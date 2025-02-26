import time
import fire

import ray
from ray.dag.input_node import InputNode
from ray.dag.output_node import MultiOutputNode
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from ray.util.accelerators import NVIDIA_A100

import torch
import logging
from ray_workers import initialize_dist_group, RayWorker


def get_logger():
    """Create and return a configured logger."""
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    dp_size_1: int = 4,
    tp_size_1: int = 1,
    dp_size_2: int = 4,
    tp_size_2: int = 1,
    batch_size: int = 16,
):
    logger = get_logger()

    # Initialize dataset
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_gen")

    def get_input(examples):
        return [msg[0]["content"] for msg in examples["messages"]]

    def split_batch(examples, worker_idx, num_workers):
        """Split batch for each worker"""
        start_idx = (worker_idx * batch_size) // num_workers
        end_idx = ((worker_idx + 1) * batch_size) // num_workers
        # Extract just the text from the messages (first message is the prompt)
        return get_input(examples)[start_idx:end_idx]

    # Initialize worker groups
    group_1_workers = [
        RayWorker.options(accelerator_type=NVIDIA_A100).remote(
            model_name, dp_size_1, tp_size_1
        )
        for _ in range(4)
    ]
    initialize_dist_group(group_1_workers)
    ray.get([worker.init_parallel_strategy.remote() for worker in group_1_workers])

    group_2_workers = [
        RayWorker.options(accelerator_type=NVIDIA_A100).remote(
            model_name, dp_size_2, tp_size_2
        )
        for _ in range(4)
    ]
    initialize_dist_group(group_2_workers)
    ray.get([worker.init_parallel_strategy.remote() for worker in group_2_workers])

    # forward pass
    with InputNode() as input_node:
        # TODO: replace with weights
        group_1_activations = [
            worker.forward.bind(input_node).with_tensor_transport("nccl")
            for worker in group_1_workers
        ]

        # # TODO: fix the weight sync graph
        group_2_outputs = [
            group_2_workers[i].recv_weights.bind(activation)
            for i, activation in enumerate(group_1_activations)
        ]
        sync_param_dag = MultiOutputNode(group_2_outputs)
    sync_param_dag = sync_param_dag.experimental_compile(_submit_timeout=5000)

    for batch_idx in range(0, len(dataset), batch_size):
        batch = dataset[batch_idx : batch_idx + batch_size]
        
        results = ray.get(sync_param_dag.execute(get_input(batch)))
        for i, result in enumerate(results):
            logger.info(f"Worker {i}: {result}")
        break  # Remove this to process more than one batch
    
    sync_param_dag.teardown()


if __name__ == "__main__":
    fire.Fire(main)
