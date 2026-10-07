"""Check an existing integer broadcast limitation separately from mixed restore."""

import re

import jax
import numpy as np
from orbax.checkpoint._src.serialization import type_handlers_test


class NarrowIntegerControl(type_handlers_test.SingleReplicaArrayHandlerTest):

  async def test_int32_broadcast_control(self):
    config = type_handlers_test.SingleReplicaTestConfig(
        mesh=jax.sharding.Mesh(
            np.asarray(jax.devices()).reshape(2, 4), ('x', 'y')
        ),
        np_arrays=[np.arange(64, dtype=np.int32).reshape(8, 8)],
        partition_specs=[jax.sharding.PartitionSpec(None, 'y')],
    )
    with self.assertRaisesRegex(
        AssertionError, re.escape("dtype('int64') != dtype('int32')")
    ):
      await self.single_replica_serialize_deserialize(config)
    print('Unmodified single-replica int32 broadcast restores int64 with x64 enabled')
