import time
import fire

import ray
from ray.dag.input_node import InputNode
from ray.dag.output_node import MultiOutputNode
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from ray.util.accelerators import NVIDIA_A100

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
    dp_size_1: int = 2,
    tp_size_1: int = 1,
    dp_size_2: int = 2,
    tp_size_2: int = 1,
    batch_size: int = 16,
):
    # Initialize worker groups
    group_1_workers = [
        RayWorker.options(accelerator_type=NVIDIA_A100).remote(
            model_name, dp_size_1, tp_size_1
        )
        for _ in range(dp_size_1)
    ]
    initialize_dist_group(group_1_workers)
    ray.get([worker.init_parallel_strategy.remote() for worker in group_1_workers])

    group_2_workers = [
        RayWorker.options(accelerator_type=NVIDIA_A100).remote(
            model_name, dp_size_2, tp_size_2
        )
        for _ in range(dp_size_2)
    ]
    initialize_dist_group(group_2_workers)
    ray.get([worker.init_parallel_strategy.remote() for worker in group_2_workers])

    # need to collect weights on rank zero worker by calling state_dict on all workers
    ray.get([worker.collect_weights.remote() for worker in group_1_workers])
    
    # forward pass
    with InputNode() as input_node:
        weights = group_1_workers[0].send_weights.bind(input_node).with_tensor_transport("nccl")        

        group_2_outputs = [
            group_2_workers[i].recv_weights.bind(weights)
            for i in range(dp_size_2)
        ]
        sync_param_dag = MultiOutputNode(group_2_outputs)
    sync_param_dag = sync_param_dag.experimental_compile(_submit_timeout=5000)
    ray.get(sync_param_dag.execute(1))
    
    sync_param_dag.teardown()


if __name__ == "__main__":
    fire.Fire(main)
