import os
import logging
import ray
import torch
import warnings
from ray.air.util.torch_dist import _init_torch_distributed
from ray.air._internal.util import find_free_port
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import List
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import functional as F
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)


def get_logger():
    """Create and return a configured logger."""
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


@dataclass
class TorchDistributedConfig:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    master_addr: str
    master_port: str
    gpu_ids: List[int]


def initialize_dist_group(workers):
    """Initialize PyTorch Distributed Process Group for a set of workers."""
    worker_metadata = ray.get([worker.get_metadata.remote() for worker in workers])

    for worker_id, metadata in enumerate(worker_metadata):
        metadata["worker_id"] = worker_id

    aggregated_metadata = defaultdict(list)

    for metadata in worker_metadata:
        aggregated_metadata[metadata["address"]].append(metadata)

    for metadata_list_per_ip in aggregated_metadata.values():
        metadata_list_per_ip.sort(key=lambda x: x["gpu_ids"])

    rank = 0
    world_size = len(workers)
    dist_configs = dict()

    for metadata_list_per_ip in aggregated_metadata.values():
        local_rank = 0
        local_world_size = len(metadata_list_per_ip)
        visible_device_ids = []

        for metadata in metadata_list_per_ip:
            visible_device_ids += metadata["gpu_ids"]

        for metadata in metadata_list_per_ip:
            if rank == 0:
                master_addr = metadata["address"]
                master_port = metadata["port"]

            worker_id = metadata["worker_id"]
            worker_config = TorchDistributedConfig(
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                local_world_size=local_world_size,
                master_addr=master_addr,
                master_port=master_port,
                gpu_ids=visible_device_ids,
            )

            rank += 1
            local_rank += 1

            dist_configs[worker_id] = worker_config

    ray.get(
        [
            worker.init_dist_group.remote(dist_configs[worker_id])
            for worker_id, worker in enumerate(workers)
        ]
    )
    print("Finished initializing distributed process group.")


class BaseWorker:
    def __init__(self) -> None:
        pass

    def get_metadata(self):
        return {
            "gpu_ids": ray.get_gpu_ids(),
            "address": ray.util.get_node_ip_address(),
            "port": find_free_port(),
        }

    def init_dist_group(self, dist_config):
        self.dist_config = dist_config
        _init_torch_distributed(
            init_method="env", backend="nccl", **asdict(dist_config)
        )
        print(f"Rank {self.dist_config.rank}: Initialized")
        if self.dist_config.rank == 0:
            print(asdict(self.dist_config))


@ray.remote(num_gpus=1)
class RayWorker(BaseWorker):
    def __init__(self, model_name, dp_size, tp_size) -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dp_size = dp_size
        self.tp_size = tp_size

        self.logger = get_logger()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=None,
            )

    def init_parallel_strategy(self):
        self.rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.rank)
        torch.cuda.reset_peak_memory_stats()

        self.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        assert (
            self.world_size == self.tp_size * self.dp_size
        ), "world size must be equal to tp size * dp size"

        if self.world_size > 1:
            self.device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(self.dp_size, self.tp_size),
                mesh_dim_names=("dp", "tp"),
            )

            self.tp_rank = self.device_mesh["tp"].get_local_rank()
            self.dp_rank = self.device_mesh["dp"].get_local_rank()
            
            if self.dp_size == 1:
                rank_log(self.dp_rank, self.tp_rank, f"using TP", self.logger)
                self.model = parallelize_tp(self.model, self.device_mesh["tp"])
            else:
                rank_log(self.dp_rank, self.tp_rank, f"using FSDP", self.logger)
                self.model = parallelize_dp(self.model, self.device_mesh["dp"])

        else:
            self.model.to("cuda")
            self.device_mesh = None
            self.tp_rank = 0
            self.dp_rank = 0

    def get_rank(self):
        return self.dp_rank, self.tp_rank

    def forward(self, batch):
        """
        Forward pass through the model
        Args:
            batch: List of text inputs
        Returns:
            logits from the model
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}
        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.logits
        
    def send_weights(self):
        raise NotImplementedError("Not implemented")
    
    def recv_weights(self, tensors):
        """simulate recv weights with receiving forward activations"""
        return tensors.size()


# TODO(anm): need to adapt this to HF transformers
def parallelize_transformer(transformer, tp_mesh):
    for _, transformer_block in enumerate(transformer.resblocks):
        layer_tp_plan = {
            # "ln_1": SequenceParallel(),
            # "attn": PrepareModuleInput(
            #     input_layouts=(Shard(1), None),
            #     desired_input_layouts=(Replicate(), None),
            # ),
            # "attn.in_proj_weight": ColwiseParallel(),
            # "attn.out_proj": RowwiseParallel(output_layouts=Shard(1)),
            # "attn.out_proj": RowwiseParallel(),
            # "ln_2": SequenceParallel(),
            # "mlp": PrepareModuleInput(
            #     input_layouts=(Shard(1),),
            #     desired_input_layouts=(Replicate(),),
            # ),
            "mlp.c_fc": ColwiseParallel(),
            # "mlp.c_proj": RowwiseParallel(output_layouts=Shard(1)),
            "mlp.c_proj": RowwiseParallel(),
        }

        # # Adjust attention module to use the local number of heads
        # attn_layer = transformer_block.attn
        # attn_layer.num_heads = attn_layer.num_heads // tp_mesh.size()

        # Custom parallelization plan for the model
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    return transformer


def parallelize_tp(model, tp_mesh):
    """
    Imitate the example in https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py
    """
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "token_embedding": RowwiseParallel(
                input_layouts=Replicate(),
            ),
        },
    )
    parallelize_transformer(model, tp_mesh)

    model.to("cuda")
    return model


def parallelize_dp(model, dp_mesh):
    model.to(torch.cuda.current_device())
    model = FSDP(
        model,
        device_mesh=dp_mesh,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )
    return model


def rank_log(dp_rank, tp_rank, msg, logger):
    """helper function to log only on all ranks"""
    logger.info(f"[dp{dp_rank}-tp{tp_rank}] {msg}")


def parallelize_2d(model, dp_size, tp_size, logger=None):
    assert dp_size * tp_size > 1, "DP or TP must be greater than 1!"

    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    dp_rank = device_mesh["dp"].get_local_rank()
    tp_rank = device_mesh["tp"].get_local_rank()

    if dp_size == 1:
        rank_log(dp_rank, tp_rank, f"using TP", logger)
        model = parallelize_tp(model, device_mesh["tp"])
    else:
        rank_log(dp_rank, tp_rank, f"using FSDP", logger)
        model = parallelize_dp(model, device_mesh["dp"])

    return model, device_mesh
