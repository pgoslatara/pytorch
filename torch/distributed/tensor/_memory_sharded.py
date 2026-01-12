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
        _padded_local: 1D flattened tensor with padding for even all-gather.
    """

    _storage_spec: StorageShardingSpec
    _process_group: dist.ProcessGroup
    _padded_local: torch.Tensor
    __slots__ = ["_storage_spec", "_process_group", "_padded_local"]

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: StorageShardingSpec,
        process_group: dist.ProcessGroup,
        padded_local: torch.Tensor,
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
        r._padded_local = padded_local
        return r

    def __init__(
        self,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: StorageShardingSpec,
        process_group: dist.ProcessGroup,
        padded_local: torch.Tensor,
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
        padded_local: Optional[torch.Tensor] = None,
    ) -> "MemoryShardedDTensor":
        """
        Factory method to create a MemoryShardedDTensor.

        Args:
            local_tensor: The local shard of the tensor on this rank.
            device_mesh: The DeviceMesh for this distributed tensor.
            storage_spec: StorageShardingSpec describing sharding configuration.
            process_group: Process group for collective operations.
            placements: DTensor placements tuple.
            padded_local: Optional pre-computed 1D padded tensor. If None,
                will be computed from local_tensor and storage_spec.

        Returns:
            A new MemoryShardedDTensor instance.
        """
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

        # Compute padded_local if not provided
        if padded_local is None:
            padded_local = cls._compute_padded_local(local_tensor, storage_spec)

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
            padded_local,
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

    @staticmethod
    def _compute_padded_local(
        local_tensor: torch.Tensor,
        storage_spec: StorageShardingSpec,
    ) -> torch.Tensor:
        """
        Compute the 1D padded tensor from a local shard and storage spec.

        Args:
            local_tensor: The local shard tensor.
            storage_spec: StorageShardingSpec with padding info.

        Returns:
            1D flattened tensor, padded to padded_shard_size * other_dims.
        """
        actual_size = storage_spec.actual_shard_size
        padded_size = storage_spec.padded_shard_size
        shard_dim = storage_spec.shard_dim

        # Calculate number of elements in other dimensions
        other_dims_numel = 1
        for i, s in enumerate(local_tensor.shape):
            if i != shard_dim:
                other_dims_numel *= s

        # Total padded numel
        padded_numel = padded_size * other_dims_numel

        if actual_size == padded_size:
            # No padding needed - just flatten
            return local_tensor.view(-1)
        else:
            # Need to pad: create padded buffer and copy data
            padded = local_tensor.new_zeros(padded_numel)
            padded[: local_tensor.numel()].copy_(local_tensor.view(-1))
            return padded

    def detach(self) -> "MemoryShardedDTensor":
        """
        Returns a detached MemoryShardedDTensor.

        This is required for nn.Parameter compatibility - the detach() method
        must return an instance of the same type.

        Returns:
            A new MemoryShardedDTensor with detached local tensor.
        """
        detached_local = self._local_tensor.detach()
        detached_padded = self._padded_local.detach()
        return self._create(
            local_tensor=detached_local,
            device_mesh=self._spec.mesh,
            storage_spec=self._storage_spec,
            process_group=self._process_group,
            placements=self._spec.placements,
            padded_local=detached_padded,
        )

    def _get_padded_local(self) -> torch.Tensor:
        """
        Returns the local tensor padded to the padded_shard_size as ND tensor.

        For uneven sharding, the last rank may have a smaller shard than the
        padded size. This method returns an ND view of the padded local tensor
        for operations that need multi-dimensional access (like unshard).

        Returns:
            Local tensor padded to padded_shard_size on the shard dimension.
        """
        spec = self._storage_spec
        local_tensor = self._local_tensor

        # Reshape _padded_local (1D) to ND with padded shard size
        padded_shape = list(local_tensor.shape)
        padded_shape[spec.shard_dim] = spec.padded_shard_size

        return self._padded_local.view(padded_shape)

    def get_all_gather_input(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Returns a 1D flattened tensor suitable for all-gather collective operations.

        This method is used by FSDP to get the input tensor for batched all-gather.
        Returns the pre-computed padded tensor for O(1) access.

        Args:
            dtype: If provided, convert the tensor to this dtype before returning.
                   Used for mixed precision training where storage dtype differs
                   from compute dtype.

        Returns:
            A 1D flattened plain torch.Tensor (not DTensor) containing the padded
            local shard, suitable for passing to all-gather collectives.

        Example:
            >>> sharded = distribute_storage(dtensor, dim=0, mesh_dim="dp")
            >>> all_gather_input = sharded.get_all_gather_input(torch.float16)
            >>> # Use all_gather_input in batched all-gather collective
        """
        # _padded_local is already 1D and padded - O(1) access
        result = self._padded_local

        # Apply dtype conversion if needed
        if dtype is not None and result.dtype != dtype:
            result = result.to(dtype)

        return result

    @classmethod
    def from_local_shard(
        cls,
        local_shard: torch.Tensor,
        full_shape: torch.Size,
        shard_dim: int,
        device_mesh: DeviceMesh,
        mesh_dim: int | str,
        *,
        requires_grad: bool = False,
        placements: tuple | None = None,
        padded_local: Optional[torch.Tensor] = None,
    ) -> "MemoryShardedDTensor":
        """
        Create a MemoryShardedDTensor from an already-sharded local tensor.

        This factory method is used by FSDP when it has already computed the
        local shard and needs to wrap it in a MemoryShardedDTensor. Unlike
        distribute_storage() which shards a full tensor, this method takes
        a pre-sharded local tensor.

        Args:
            local_shard: The local shard tensor on this rank.
            full_shape: The original (full) shape of the tensor before sharding.
            shard_dim: The dimension along which the tensor is sharded.
            device_mesh: The DeviceMesh for this distributed tensor.
            mesh_dim: The mesh dimension (name or index) used for sharding.
            requires_grad: Whether the tensor requires gradient computation.
            placements: Optional DTensor placements tuple. If None, defaults to
                all-Replicate placements. For TP+FSDP case, pass the combined
                SPMD placements (e.g., (Shard(dim), Shard(dim)) for TP sharding).
            padded_local: Optional pre-computed 1D padded tensor. If None,
                will be computed from local_shard.

        Returns:
            A MemoryShardedDTensor wrapping the local shard.

        Example:
            >>> # FSDP has already computed the local shard
            >>> local_shard = full_param.narrow(0, start, length).contiguous()
            >>> sharded = MemoryShardedDTensor.from_local_shard(
            ...     local_shard=local_shard,
            ...     full_shape=full_param.shape,
            ...     shard_dim=0,
            ...     device_mesh=mesh,
            ...     mesh_dim="dp",
            ... )
        """
        from torch.distributed.tensor.placement_types import Replicate

        # Resolve mesh_dim to index and name
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

        # Get process group and world size
        process_group = device_mesh.get_group(mesh_dim_idx)
        world_size = device_mesh.size(mesh_dim_idx)

        # Compute padded shard size from full shape
        full_size_on_dim = full_shape[shard_dim]
        padded_shard_size = (full_size_on_dim + world_size - 1) // world_size

        # Actual shard size is the size of the local tensor on shard_dim
        actual_shard_size = local_shard.size(shard_dim)

        # Compute original stride (assume contiguous layout for full tensor)
        orig_stride = []
        stride = 1
        for i in range(len(full_shape) - 1, -1, -1):
            orig_stride.insert(0, stride)
            stride *= full_shape[i]
        orig_stride = tuple(orig_stride)

        # Create storage sharding spec
        storage_spec = StorageShardingSpec(
            orig_size=full_shape,
            orig_stride=orig_stride,
            shard_dim=shard_dim,
            mesh_dim=mesh_dim_name,
            padded_shard_size=padded_shard_size,
            actual_shard_size=actual_shard_size,
        )

        # Use provided placements or default to all-Replicate
        if placements is None:
            placements = tuple(Replicate() for _ in range(device_mesh.ndim))

        # Ensure requires_grad is set correctly
        if requires_grad and not local_shard.requires_grad:
            local_shard = local_shard.requires_grad_(True)

        return cls._create(
            local_tensor=local_shard,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=process_group,
            placements=placements,
            padded_local=padded_local,
        )

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
        raise ValueError(f"dim {dim} is out of range for tensor with {ndim} dimensions")

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
        mesh_dim_name = mesh_dim_names[mesh_dim_idx] if mesh_dim_names else "default"

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
