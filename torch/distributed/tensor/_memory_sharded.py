# Copyright (c) Meta Platforms, Inc. and affiliates
"""
MemoryShardedDTensor: A DTensor variant that shards storage across devices.

This module provides a memory-efficient DTensor implementation where the tensor's
storage is sharded across devices in a process group, reducing per-device memory
usage. Unlike regular DTensor sharding which affects the logical tensor view,
MemoryShardedDTensor physically partitions the underlying storage.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


@dataclass
class StorageShardingSpec:
    """
    Specification for how a tensor's storage is sharded across devices.

    Attributes:
        orig_size: Original (full) tensor size before sharding.
        orig_stride: Original tensor stride before sharding.
        shard_dim: The tensor dimension along which storage is sharded.
        mesh_dim: The mesh dimension name used for sharding.
        padded_shard_size: Size of each shard after padding for even division.
        actual_shard_size: Actual size of the shard on this rank (may be smaller
            than padded_shard_size for the last rank with uneven sharding).
    """

    orig_size: torch.Size
    orig_stride: tuple[int, ...]
    shard_dim: int
    mesh_dim: str
    padded_shard_size: int
    actual_shard_size: int


class MemoryShardedDTensor(DTensor):
    """
    A DTensor subclass that physically shards storage across devices.

    MemoryShardedDTensor reduces per-device memory by partitioning the tensor's
    storage along a specified dimension. Each device holds only its local shard.
    The full tensor can be reconstructed via the unshard() method which performs
    an all-gather collective.

    This is useful for FSDP-style memory savings where parameters are sharded
    during forward/backward and gathered only when needed.

    Attributes:
        _storage_spec: StorageShardingSpec describing the sharding configuration.
        _process_group: The process group used for collective operations.
    """

    _storage_spec: StorageShardingSpec
    _process_group: dist.ProcessGroup
    __slots__ = ["_storage_spec", "_process_group"]

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: StorageShardingSpec,
        process_group: dist.ProcessGroup,
        *,
        requires_grad: bool,
    ) -> "MemoryShardedDTensor":
        # Create the DTensor base
        r = super().__new__(
            cls,
            local_tensor,
            spec,
            requires_grad=requires_grad,
        )
        r._storage_spec = storage_spec
        r._process_group = process_group
        return r

    def __init__(
        self,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: StorageShardingSpec,
        process_group: dist.ProcessGroup,
        *,
        requires_grad: bool,
    ) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return (
            f"MemoryShardedDTensor(local_shape={self.shape}, "
            f"full_shape={self.full_shape}, "
            f"shard_dim={self._storage_spec.shard_dim}, "
            f"device_mesh={self._spec.mesh})"
        )

    @classmethod
    def _create(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        storage_spec: StorageShardingSpec,
        process_group: dist.ProcessGroup,
        placements: tuple,
    ) -> "MemoryShardedDTensor":
        """
        Factory method to create a MemoryShardedDTensor.

        Args:
            local_tensor: The local shard of the tensor on this rank.
            device_mesh: The DeviceMesh for this distributed tensor.
            storage_spec: StorageShardingSpec describing sharding configuration.
            process_group: Process group for collective operations.
            placements: DTensor placements tuple.

        Returns:
            A new MemoryShardedDTensor instance.
        """
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

        # Build DTensorSpec with the local tensor's metadata
        tensor_meta = TensorMeta(
            shape=local_tensor.shape,
            stride=local_tensor.stride(),
            dtype=local_tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return cls(
            local_tensor,
            dtensor_spec,
            storage_spec,
            process_group,
            requires_grad=local_tensor.requires_grad,
        )

    @property
    def full_shape(self) -> torch.Size:
        """
        Returns the original (full) shape of the tensor before sharding.
        """
        return self._storage_spec.orig_size

    def full_size(self, dim: Optional[int] = None) -> int | torch.Size:
        """
        Returns the original (full) size of the tensor.

        Args:
            dim: If specified, returns the size of that dimension.
                 If None, returns the full shape.

        Returns:
            Size of the specified dimension, or full shape if dim is None.
        """
        if dim is None:
            return self.full_shape
        return self._storage_spec.orig_size[dim]

    def local(self) -> torch.Tensor:
        """
        Returns the local shard as a torch.Tensor.

        Returns:
            The underlying local tensor shard.
        """
        return self._local_tensor

    @property
    def storage_spec(self) -> StorageShardingSpec:
        """
        Returns the storage sharding specification.
        """
        return self._storage_spec

    @property
    def process_group(self) -> dist.ProcessGroup:
        """
        Returns the process group used for collective operations.
        """
        return self._process_group

    def _get_padded_local(self) -> torch.Tensor:
        """
        Returns the local tensor padded to the padded_shard_size if needed.

        For uneven sharding, the last rank may have a smaller shard than the
        padded size. This method pads the local tensor with zeros to ensure
        all ranks have the same size for the all-gather operation.

        Returns:
            Local tensor padded to padded_shard_size on the shard dimension.
        """
        spec = self._storage_spec
        local_tensor = self._local_tensor

        if spec.actual_shard_size == spec.padded_shard_size:
            # No padding needed
            return local_tensor

        # Need to pad the local tensor
        shard_dim = spec.shard_dim
        pad_size = spec.padded_shard_size - spec.actual_shard_size

        # Create padding shape
        pad_shape = list(local_tensor.shape)
        pad_shape[shard_dim] = pad_size

        # Create zero padding
        padding = local_tensor.new_zeros(pad_shape)

        # Concatenate along shard dimension
        return torch.cat([local_tensor, padding], dim=shard_dim)

    def unshard(self) -> DTensor:
        """
        Reconstruct the full tensor via all-gather collective.

        Performs an all-gather operation to collect all shards from all ranks
        in the process group, then reconstructs the original tensor shape.

        Returns:
            A DTensor containing the full (unsharded) tensor data replicated
            across all ranks.

        Example:
            >>> sharded = distribute_storage(dtensor, dim=0, mesh_dim="dp")
            >>> sharded.shape  # (4, 8) - local shard
            >>> full = sharded.unshard()
            >>> full.shape  # (16, 8) - full tensor
        """
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        spec = self._storage_spec
        world_size = dist.get_world_size(self._process_group)

        # Get padded local tensor for even all-gather
        padded_local = self._get_padded_local()

        # Detach for all-gather to avoid autograd issues, track original requires_grad
        orig_requires_grad = padded_local.requires_grad
        if orig_requires_grad:
            padded_local = padded_local.detach()

        # Compute output shape for all-gather (gathered along shard dim)
        gathered_shape = list(padded_local.shape)
        gathered_shape[spec.shard_dim] = spec.padded_shard_size * world_size

        # Allocate output tensor
        gathered_tensor = padded_local.new_empty(gathered_shape)

        # Reshape for all_gather_into_tensor which expects flat gather dimension
        # We need to gather into a contiguous buffer, then reshape
        # all_gather_into_tensor gathers along dimension 0, so we need to handle
        # multi-dimensional tensors carefully

        # Flatten approach: move shard_dim to front, then gather
        shard_dim = spec.shard_dim
        ndim = padded_local.ndim

        if shard_dim != 0:
            # Move shard_dim to position 0
            perm = [shard_dim] + [i for i in range(ndim) if i != shard_dim]
            padded_local = padded_local.permute(perm).contiguous()

        # Now shard dimension is at position 0
        # All-gather expects to gather world_size copies
        input_flat = padded_local
        output_flat = padded_local.new_empty(
            (world_size * input_flat.size(0),) + input_flat.shape[1:]
        )

        # Perform all-gather
        dist.all_gather_into_tensor(
            output_flat,
            input_flat,
            group=self._process_group,
        )

        # Reshape back: output_flat has shape [world_size * padded_shard_size, ...]
        # Need to reconstruct [padded_shard_size * world_size, original_other_dims]
        # and then permute back if needed

        if shard_dim != 0:
            # We have [world_size * shard, d1, d2, ...] but dims are permuted
            # Need to get back to original order
            # Create inverse permutation
            inv_perm = [0] * ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            output_flat = output_flat.permute(inv_perm).contiguous()

        # Slice to original size (remove padding)
        orig_size_on_shard = spec.orig_size[shard_dim]
        gathered_tensor = output_flat.narrow(shard_dim, 0, orig_size_on_shard)

        # Restore original stride via as_strided if needed, or just ensure contiguous
        gathered_tensor = gathered_tensor.contiguous()

        # Preserve requires_grad
        if orig_requires_grad:
            gathered_tensor = gathered_tensor.requires_grad_(True)

        # Create DTensor with Replicate placement
        device_mesh = self._spec.mesh
        placements = tuple(Replicate() for _ in range(device_mesh.ndim))

        tensor_meta = TensorMeta(
            shape=gathered_tensor.shape,
            stride=gathered_tensor.stride(),
            dtype=gathered_tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return DTensor(
            gathered_tensor,
            dtensor_spec,
            requires_grad=gathered_tensor.requires_grad,
        )


def distribute_storage(
    dtensor: DTensor,
    dim: int,
    mesh_dim: int | str,
) -> MemoryShardedDTensor:
    """
    Create a MemoryShardedDTensor by sharding a DTensor's storage along a dimension.

    This function takes a DTensor and shards its underlying storage along the
    specified dimension across devices in the given mesh dimension. Unlike
    DTensor's logical sharding, this physically partitions the storage to
    reduce per-device memory usage.

    Args:
        dtensor: The input DTensor to shard. Must be replicated on the target
            mesh dimension.
        dim: The tensor dimension along which to shard storage. Must be in
            range [-ndim, ndim).
        mesh_dim: The mesh dimension (name or index) to use for sharding.

    Returns:
        A MemoryShardedDTensor with storage sharded across devices.

    Raises:
        ValueError: If dim is out of range or mesh_dim doesn't exist.

    Example:
        >>> # FSDP-style sharding: shard parameters along dim 0
        >>> mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        >>> param = distribute_tensor(torch.randn(16, 8), mesh, [Replicate()])
        >>> sharded = distribute_storage(param, dim=0, mesh_dim="dp")
        >>> sharded.shape  # Local shape: (4, 8)
        >>> sharded.full_shape  # Original shape: (16, 8)
    """
    from torch.distributed.tensor.placement_types import Replicate

    device_mesh = dtensor.device_mesh
    ndim = dtensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = dim + ndim

    # Validate dim is in range
    if dim < 0 or dim >= ndim:
        raise ValueError(
            f"dim {dim} is out of range for tensor with {ndim} dimensions"
        )

    # Resolve mesh_dim to index if it's a string
    if isinstance(mesh_dim, str):
        mesh_dim_names = device_mesh.mesh_dim_names
        if mesh_dim_names is None or mesh_dim not in mesh_dim_names:
            raise ValueError(
                f"mesh_dim '{mesh_dim}' not found in device mesh. "
                f"Available dimensions: {mesh_dim_names}"
            )
        mesh_dim_name = mesh_dim
        mesh_dim_idx = mesh_dim_names.index(mesh_dim)
    else:
        mesh_dim_idx = mesh_dim
        if mesh_dim_idx < 0 or mesh_dim_idx >= device_mesh.ndim:
            raise ValueError(
                f"mesh_dim {mesh_dim_idx} is out of range for mesh with "
                f"{device_mesh.ndim} dimensions"
            )
        mesh_dim_names = device_mesh.mesh_dim_names
        mesh_dim_name = (
            mesh_dim_names[mesh_dim_idx] if mesh_dim_names else "default"
        )

    # Get process group and world size for the mesh dimension
    process_group = device_mesh.get_group(mesh_dim_idx)
    world_size = device_mesh.size(mesh_dim_idx)
    local_rank = device_mesh.get_local_rank(mesh_dim_idx)

    # Get the full tensor data (replicated on all ranks)
    full_tensor = dtensor.to_local()

    # Compute shard sizes
    full_size_on_dim = full_tensor.size(dim)
    # Use ceiling division for padded shard size
    padded_shard_size = (full_size_on_dim + world_size - 1) // world_size

    # Compute actual shard size for this rank
    start_idx = local_rank * padded_shard_size
    end_idx = min(start_idx + padded_shard_size, full_size_on_dim)
    actual_shard_size = max(0, end_idx - start_idx)

    # Extract the local shard
    if actual_shard_size > 0:
        local_shard = full_tensor.narrow(dim, start_idx, actual_shard_size)
        # Make contiguous copy to own the storage
        local_shard = local_shard.contiguous()
    else:
        # Empty shard for ranks beyond the tensor size
        shard_shape = list(full_tensor.shape)
        shard_shape[dim] = 0
        local_shard = full_tensor.new_empty(shard_shape)

    # Preserve requires_grad
    if full_tensor.requires_grad:
        local_shard = local_shard.requires_grad_(True)

    # Create storage sharding spec
    storage_spec = StorageShardingSpec(
        orig_size=full_tensor.size(),
        orig_stride=full_tensor.stride(),
        shard_dim=dim,
        mesh_dim=mesh_dim_name,
        padded_shard_size=padded_shard_size,
        actual_shard_size=actual_shard_size,
    )

    # Create placements - replicated on all dimensions
    placements = tuple(Replicate() for _ in range(device_mesh.ndim))

    return MemoryShardedDTensor._create(
        local_tensor=local_shard,
        device_mesh=device_mesh,
        storage_spec=storage_spec,
        process_group=process_group,
        placements=placements,
    )
