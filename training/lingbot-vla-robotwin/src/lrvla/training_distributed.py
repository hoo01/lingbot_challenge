"""Small torchrun/DDP primitives kept separate from the training policy."""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool = False

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    @classmethod
    def from_environment(cls, requested_device="cuda", *, dry_run=False):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
            raise ValueError("Invalid torchrun WORLD_SIZE/RANK/LOCAL_RANK environment.")
        requested = torch.device(requested_device)
        if world_size > 1 and requested.type != "cuda":
            raise ValueError("Multi-process training requires CUDA/NCCL.")
        if requested.type == "cuda" and not torch.cuda.is_available() and not dry_run:
            raise RuntimeError("CUDA unavailable in this environment.")
        initialized_here = False
        if world_size > 1:
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible.")
            torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
                initialized_here = True
            device = torch.device("cuda", local_rank)
        elif requested.type == "cuda" and torch.cuda.is_available():
            index = requested.index if requested.index is not None else 0
            torch.cuda.set_device(index)
            device = torch.device("cuda", index)
        elif requested.type == "cuda" and dry_run:
            device = requested
        else:
            device = requested
        return cls(rank, local_rank, world_size, device, initialized_here)

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def broadcast_object(self, value, source=0):
        if not self.distributed:
            return value
        values = [value if self.rank == source else None]
        dist.broadcast_object_list(values, src=source)
        return values[0]

    def sum_scalar(self, value):
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def max_scalar(self, value):
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def all_true(self, value):
        tensor = torch.tensor(1 if value else 0, dtype=torch.int32, device=self.device)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(tensor.item())

    def gather_objects(self, value):
        if not self.distributed:
            return [value]
        output = [None] * self.world_size if self.is_main else None
        dist.gather_object(value, output, dst=0)
        return output

    def close(self):
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def distributed_metadata(context, per_device_batch, accumulation):
    return {
        "backend": "nccl" if context.distributed else None,
        "world_size": context.world_size,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation": accumulation,
        "global_batch_size": per_device_batch * accumulation * context.world_size,
        "sampler": "global_shuffle_rank_slice_drop_last" if context.distributed else "epoch_batch_sampler",
    }
