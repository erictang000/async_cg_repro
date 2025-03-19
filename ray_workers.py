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
    def __init__(self, model_name, dp_size) -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dp_size = dp_size

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
            self.world_size == self.dp_size
        ), "world size must be equal to dp size"

        if self.world_size > 1:
            self.device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(self.dp_size,),
                mesh_dim_names=("dp",),
            )

            self.dp_rank = self.device_mesh["dp"].get_local_rank()
    
            rank_log(self.dp_rank, f"using FSDP", self.logger)
            self.model = parallelize_dp(self.model, self.device_mesh["dp"])
        else:
            self.model.to("cuda")
            self.device_mesh = None
            self.dp_rank = 0

    def get_rank(self):
        return self.dp_rank
    
    async def test_async(self):
        print(f"role {self.role} rank {self.rank} testing async")
        await asyncio.sleep(2)
        print(f"role {self.role} rank {self.rank} done testing async")

    def collect_weights(self):
        if self.world_size > 1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                self.full_state_dict = self.model.state_dict()
        else:
            self.full_state_dict = self.model.state_dict()

    def send_weights(self, x):
        for key, weight in self.full_state_dict.items():
            assert weight.device.index == self.rank
        return self.full_state_dict
    
    def recv_weights(self, state_dict):
        """simulate recv weights with receiving forward activations"""
        for key, weight in state_dict.items():
            assert weight.device.index == self.rank
        print("yay!")

def parallelize_dp(model, dp_mesh):
    model.to(torch.cuda.current_device())
    model = FSDP(
        model,
        device_mesh=dp_mesh,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )
    return model


def rank_log(dp_rank, msg, logger):
    """helper function to log only on all ranks"""
    logger.info(f"[dp{dp_rank}] {msg}")
