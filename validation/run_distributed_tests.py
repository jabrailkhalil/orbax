"""Run Orbax's existing test classes on two CPU processes, four devices each."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


def worker(rank, address, tempdir, test_args):
  from absl import flags
  import jax
  import pytest
  from orbax.checkpoint._src.testing import multiprocess_test

  flags.FLAGS(['distributed_tests', f'--test_tmpdir={tempdir}'])
  jax.distributed.initialize(
      coordinator_address=address,
      num_processes=2,
      process_id=rank,
      initialization_timeout=90,
      heartbeat_timeout_seconds=90,
  )
  print(f'Rank {rank}: {jax.devices()}', flush=True)
  try:
    result = pytest.main(['--import-mode=importlib', '-q', *test_args])
  finally:
    jax.distributed.shutdown()
  raise SystemExit(result)


def main():
  if sys.argv[1] == '--worker':
    worker(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:])
    return
  logdir = Path(os.environ.get('TEST_LOGDIR', 'distributed-test-logs'))
  logdir.mkdir(parents=True, exist_ok=True)
  with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    address = f'127.0.0.1:{sock.getsockname()[1]}'
  env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=4')
  env['JAX_PLATFORMS'] = 'cpu'
  for name in list(env):
    if name.lower().endswith('_proxy'):
      env.pop(name)
  with tempfile.TemporaryDirectory(prefix='orbax-distributed-') as tempdir:
    workers = []
    streams = []
    try:
      for rank in range(2):
        stream = (logdir / f'rank-{rank}.log').open('w')
        streams.append(stream)
        workers.append(subprocess.Popen(
            [sys.executable, __file__, '--worker', str(rank), address,
             tempdir, *sys.argv[1:]],
            env=env, stdout=stream, stderr=subprocess.STDOUT,
        ))
      results = [process.wait(timeout=1200) for process in workers]
    finally:
      for process in workers:
        if process.poll() is None:
          process.kill()
          process.wait()
      for stream in streams:
        stream.close()
      for rank in range(2):
        path = logdir / f'rank-{rank}.log'
        if path.exists():
          print(f'Rank {rank}\n{path.read_text()}')
    raise SystemExit(0 if results == [0, 0] else 1)


if __name__ == '__main__':
  main()
