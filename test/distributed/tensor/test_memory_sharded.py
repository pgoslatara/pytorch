# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Tests for MemoryShardedDTensor core class functionality.
"""
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor, init_device_mesh, Replicate
from torch.distributed.tensor._memory_sharded import (
    distribute_storage,
    MemoryShardedDTensor,
    StorageShardingSpec,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestStorageShardingSpec(TestCase):
    """Unit tests for StorageShardingSpec dataclass."""

    def test_storage_sharding_spec_creation(self):
        """Test that StorageShardingSpec can be created with all fields."""
        spec = StorageShardingSpec(
            orig_size=torch.Size([16, 32]),
            orig_stride=(32, 1),
            shard_dim=0,
            mesh_dim="dp",
            padded_shard_size=4,
            actual_shard_size=4,
        )
        self.assertEqual(spec.orig_size, torch.Size([16, 32]))
        self.assertEqual(spec.orig_stride, (32, 1))
        self.assertEqual(spec.shard_dim, 0)
        self.assertEqual(spec.mesh_dim, "dp")
        self.assertEqual(spec.padded_shard_size, 4)
        self.assertEqual(spec.actual_shard_size, 4)

    def test_storage_sharding_spec_uneven(self):
        """Test StorageShardingSpec with uneven sharding (different actual vs padded)."""
        spec = StorageShardingSpec(
            orig_size=torch.Size([13, 32]),
            orig_stride=(32, 1),
            shard_dim=0,
            mesh_dim="dp",
            padded_shard_size=4,  # ceiling(13/4) = 4
            actual_shard_size=1,  # last rank gets only 1 row
        )
        self.assertEqual(spec.padded_shard_size, 4)
        self.assertEqual(spec.actual_shard_size, 1)


class TestMemoryShardedDTensor(DTensorTestBase):
    """Distributed tests for MemoryShardedDTensor class."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_memory_sharded_dtensor_creation(self):
        """Verify MemoryShardedDTensor can be instantiated."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertIsInstance(msdt, DTensor)

    @with_comms
    def test_shape_returns_local(self):
        """Test that .shape returns local sharded shape."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.shape, torch.Size([4, 8]))

    @with_comms
    def test_full_shape_returns_original(self):
        """Test that .full_shape returns original size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))

    @with_comms
    def test_size_with_dim(self):
        """Test that .size(dim) returns local size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.size(0), 4)
        self.assertEqual(msdt.size(1), 8)

    @with_comms
    def test_full_size_with_dim(self):
        """Test that .full_size(dim) returns original size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.full_size(0), 16)
        self.assertEqual(msdt.full_size(1), 8)
        self.assertEqual(msdt.full_size(), torch.Size([16, 8]))

    @with_comms
    def test_local_returns_tensor(self):
        """Test that .local() returns torch.Tensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        result = msdt.local()
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([4, 8]))

    @with_comms
    def test_ndim(self):
        """Test that .ndim is correct."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.ndim, 2)

    @with_comms
    def test_dtype(self):
        """Test that .dtype is preserved."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type, dtype=torch.float16)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.dtype, torch.float16)

    @with_comms
    def test_device(self):
        """Test that .device is correct."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.device.type, self.device_type)

    @with_comms
    def test_requires_grad(self):
        """Test that requires_grad is preserved."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type, requires_grad=True)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertTrue(msdt.requires_grad)

        # Test with requires_grad=False
        local_tensor_no_grad = torch.randn(4, 8, device=self.device_type)
        msdt_no_grad = MemoryShardedDTensor._create(
            local_tensor=local_tensor_no_grad,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )
        self.assertFalse(msdt_no_grad.requires_grad)

    @with_comms
    def test_storage_spec_property(self):
        """Test that storage_spec property returns the spec."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.storage_spec, storage_spec)
        self.assertEqual(msdt.storage_spec.shard_dim, 0)
        self.assertEqual(msdt.storage_spec.mesh_dim, "default")

    @with_comms
    def test_process_group_property(self):
        """Test that process_group property returns the PG."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = StorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dim=0,
            mesh_dim="default",
            padded_shard_size=4,
            actual_shard_size=4,
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.process_group, pg)


class TestDistributeStorage(DTensorTestBase):
    """Distributed tests for distribute_storage() factory function."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_distribute_storage_basic(self):
        """Test basic distribute_storage creates MemoryShardedDTensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a replicated DTensor
        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard storage along dimension 0
        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))
        self.assertEqual(msdt.shape[0], 4)  # 16 / 4 ranks = 4 per rank
        self.assertEqual(msdt.shape[1], 8)

    @with_comms
    def test_distribute_storage_dim0(self):
        """Test sharding on dimension 0 (FSDP pattern)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(20, 10, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.storage_spec.shard_dim, 0)
        self.assertEqual(msdt.full_size(0), 20)
        # 20 / 4 = 5 per rank
        self.assertEqual(msdt.size(0), 5)

    @with_comms
    def test_distribute_storage_dim1(self):
        """Test sharding on dimension 1 (TP pattern)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(10, 20, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=1, mesh_dim=0)

        self.assertEqual(msdt.storage_spec.shard_dim, 1)
        self.assertEqual(msdt.full_size(1), 20)
        # 20 / 4 = 5 per rank
        self.assertEqual(msdt.size(1), 5)

    @with_comms
    def test_distribute_storage_2d_mesh(self):
        """Test distribute_storage with 2D mesh using mesh_dim name."""
        # Create 2D mesh: 2x2
        device_mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("dp", "tp")
        )

        full_tensor = torch.randn(8, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate(), Replicate()])

        # Shard on dp dimension (size 2)
        msdt = distribute_storage(dtensor, dim=0, mesh_dim="dp")

        self.assertEqual(msdt.storage_spec.mesh_dim, "dp")
        self.assertEqual(msdt.full_size(0), 8)
        # 8 / 2 (dp world size) = 4 per rank
        self.assertEqual(msdt.size(0), 4)

    @with_comms
    def test_distribute_storage_validation_dim(self):
        """Test error for invalid dim."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        with self.assertRaises(ValueError):
            distribute_storage(dtensor, dim=5, mesh_dim=0)  # Out of range

    @with_comms
    def test_distribute_storage_validation_mesh_dim(self):
        """Test error for invalid mesh_dim."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        with self.assertRaises(ValueError):
            distribute_storage(dtensor, dim=0, mesh_dim="invalid")  # Invalid name

    @with_comms
    def test_distribute_storage_negative_dim(self):
        """Test that negative dim is normalized correctly."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # dim=-1 should be equivalent to dim=1
        msdt = distribute_storage(dtensor, dim=-1, mesh_dim=0)

        self.assertEqual(msdt.storage_spec.shard_dim, 1)
        self.assertEqual(msdt.full_size(1), 8)

    @with_comms
    def test_uneven_sharding_dim0(self):
        """Test uneven sharding: 13 / 4 = uneven chunks."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 13 rows, 4 ranks: ceil(13/4) = 4 padded shard size
        # Ranks get: 4, 4, 4, 1 rows
        full_tensor = torch.randn(13, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.full_size(0), 13)
        self.assertEqual(msdt.storage_spec.padded_shard_size, 4)

        rank = dist.get_rank()
        if rank < 3:
            self.assertEqual(msdt.size(0), 4)
            self.assertEqual(msdt.storage_spec.actual_shard_size, 4)
        else:
            # Last rank gets only 1 row
            self.assertEqual(msdt.size(0), 1)
            self.assertEqual(msdt.storage_spec.actual_shard_size, 1)

    @with_comms
    def test_uneven_sharding_data_integrity(self):
        """Test that data is correctly sharded with uneven sharding."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create tensor with known values
        full_tensor = torch.arange(13 * 8, device=self.device_type).reshape(13, 8).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        rank = dist.get_rank()
        local_data = msdt.local()

        # Verify each rank has the correct slice of data
        expected_start = rank * 4  # padded_shard_size = 4
        expected_end = min(expected_start + 4, 13)
        expected_data = full_tensor[expected_start:expected_end]

        self.assertEqual(local_data.shape, expected_data.shape)
        self.assertTrue(torch.equal(local_data, expected_data))


class TestUnshard(DTensorTestBase):
    """Distributed tests for unshard() method."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_unshard_basic(self):
        """Test that unshard() returns DTensor with full data."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)
        unsharded = msdt.unshard()

        self.assertIsInstance(unsharded, DTensor)
        self.assertEqual(unsharded.shape, torch.Size([16, 8]))

    @with_comms
    def test_unshard_shape(self):
        """Test that unsharded DTensor has original shape."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(20, 10, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        # Verify sharded shape
        self.assertEqual(msdt.size(0), 5)  # 20 / 4 = 5

        # Unshard and verify original shape
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([20, 10]))

    @with_comms
    def test_unshard_data_correctness(self):
        """Test that data matches original after unshard."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create tensor with known values
        full_tensor = torch.arange(16 * 8, device=self.device_type).reshape(16, 8).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)
        unsharded = msdt.unshard()

        # Verify data matches original
        unsharded_local = unsharded.to_local()
        self.assertTrue(torch.equal(unsharded_local, full_tensor))

    @with_comms
    def test_unshard_dim1(self):
        """Test unshard on dimension 1."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.arange(10 * 20, device=self.device_type).reshape(10, 20).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard on dim 1
        msdt = distribute_storage(dtensor, dim=1, mesh_dim=0)

        self.assertEqual(msdt.size(1), 5)  # 20 / 4 = 5

        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([10, 20]))

        # Verify data
        unsharded_local = unsharded.to_local()
        self.assertTrue(torch.equal(unsharded_local, full_tensor))

    @with_comms
    def test_unshard_preserves_requires_grad(self):
        """Test that requires_grad is preserved through unshard."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type, requires_grad=True)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)
        self.assertTrue(msdt.requires_grad)

        unsharded = msdt.unshard()
        self.assertTrue(unsharded.requires_grad)

    @with_comms
    def test_unshard_uneven_sharding(self):
        """Test unshard with uneven sharding (13 / 4 = uneven)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 13 rows: ceil(13/4) = 4 per shard, ranks get 4, 4, 4, 1
        full_tensor = torch.arange(13 * 8, device=self.device_type).reshape(13, 8).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)
        unsharded = msdt.unshard()

        # Verify shape is original
        self.assertEqual(unsharded.shape, torch.Size([13, 8]))

        # Verify data matches original
        unsharded_local = unsharded.to_local()
        self.assertTrue(torch.equal(unsharded_local, full_tensor))

    @with_comms
    def test_unshard_3d_tensor(self):
        """Test unshard with 3D tensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.arange(8 * 4 * 6, device=self.device_type).reshape(8, 4, 6).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard on middle dimension
        msdt = distribute_storage(dtensor, dim=1, mesh_dim=0)

        # 4 / 4 = 1 per rank
        self.assertEqual(msdt.size(1), 1)

        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([8, 4, 6]))

        # Verify data
        unsharded_local = unsharded.to_local()
        self.assertTrue(torch.equal(unsharded_local, full_tensor))


class TestEdgeCases(DTensorTestBase):
    """Edge case tests for MemoryShardedDTensor."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_1d_tensor(self):
        """Test distribute_storage works with 1D tensors."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.arange(16, device=self.device_type).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.ndim, 1)
        self.assertEqual(msdt.size(0), 4)  # 16 / 4 = 4
        self.assertEqual(msdt.full_size(0), 16)

        # Unshard and verify
        unsharded = msdt.unshard()
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_small_tensor_fewer_than_world_size(self):
        """Test tensor smaller than world_size (some ranks get no data)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Only 3 rows, 4 ranks - rank 3 gets nothing
        full_tensor = torch.arange(3 * 4, device=self.device_type).reshape(3, 4).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        rank = dist.get_rank()
        if rank < 3:
            self.assertEqual(msdt.size(0), 1)
        else:
            self.assertEqual(msdt.size(0), 0)

        # Unshard should still work
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([3, 4]))
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_various_dtypes_float16(self):
        """Test with float16 dtype."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type, dtype=torch.float16)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.dtype, torch.float16)

        unsharded = msdt.unshard()
        self.assertEqual(unsharded.dtype, torch.float16)
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_various_dtypes_bfloat16(self):
        """Test with bfloat16 dtype."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type, dtype=torch.bfloat16)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.dtype, torch.bfloat16)

        unsharded = msdt.unshard()
        self.assertEqual(unsharded.dtype, torch.bfloat16)
        # Use allclose for bfloat16 due to precision
        self.assertTrue(torch.allclose(unsharded.to_local(), full_tensor))

    @with_comms
    def test_contiguous(self):
        """Test that sharded tensor local data is contiguous."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        self.assertTrue(msdt.local().is_contiguous())


class TestIntegration(DTensorTestBase):
    """Integration tests for MemoryShardedDTensor."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_roundtrip_dim0(self):
        """Test distribute_storage -> unshard roundtrip on dim 0."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.arange(16 * 8, device=self.device_type).reshape(16, 8).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard
        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)
        self.assertEqual(msdt.shape, torch.Size([4, 8]))

        # Unshard
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([16, 8]))

        # Verify data matches original
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_roundtrip_dim1(self):
        """Test distribute_storage -> unshard roundtrip on dim 1."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.arange(8 * 16, device=self.device_type).reshape(8, 16).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard on dim 1
        msdt = distribute_storage(dtensor, dim=1, mesh_dim=0)
        self.assertEqual(msdt.shape, torch.Size([8, 4]))  # 16 / 4 = 4

        # Unshard
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([8, 16]))

        # Verify data matches original
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_multiple_unshard(self):
        """Test that unshard can be called multiple times."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.arange(16 * 8, device=self.device_type).reshape(16, 8).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        # Unshard multiple times
        unsharded1 = msdt.unshard()
        unsharded2 = msdt.unshard()

        # Both should have correct data
        self.assertTrue(torch.equal(unsharded1.to_local(), full_tensor))
        self.assertTrue(torch.equal(unsharded2.to_local(), full_tensor))

    @with_comms
    def test_different_mesh_dim_names(self):
        """Test with various mesh dimension names."""
        device_mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("data_parallel", "model_parallel")
        )

        full_tensor = torch.arange(8 * 8, device=self.device_type).reshape(8, 8).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate(), Replicate()])

        # Shard on "data_parallel" dimension
        msdt = distribute_storage(dtensor, dim=0, mesh_dim="data_parallel")

        self.assertEqual(msdt.storage_spec.mesh_dim, "data_parallel")
        self.assertEqual(msdt.size(0), 4)  # 8 / 2 = 4

        # Unshard
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([8, 8]))
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_roundtrip_uneven(self):
        """Test roundtrip with uneven sharding."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 15 rows, 4 ranks: ceil(15/4) = 4 per shard
        # Ranks get: 4, 4, 4, 3
        full_tensor = torch.arange(15 * 8, device=self.device_type).reshape(15, 8).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = distribute_storage(dtensor, dim=0, mesh_dim=0)

        # Verify uneven distribution
        rank = dist.get_rank()
        if rank < 3:
            self.assertEqual(msdt.size(0), 4)
        else:
            self.assertEqual(msdt.size(0), 3)

        # Unshard and verify
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([15, 8]))
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))


if __name__ == "__main__":
    run_tests()
