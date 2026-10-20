"""Run the single-host checkpoint suite with upstream's multiprocess exclusions."""

from pathlib import Path
import subprocess
import sys

import yaml

checkpoint = Path('checkpoint')
extra_ignores = [
    'orbax/checkpoint/experimental/emergency/broadcast_multislice_test.py',
    'orbax/checkpoint/experimental/emergency/checkpoint_manager_test.py',
    'orbax/checkpoint/experimental/emergency/single_slice_checkpoint_manager_test.py',
    'orbax/checkpoint/experimental/emergency/local_checkpoint_data_debugging_test.py',
    'orbax/checkpoint/experimental/emergency/local_checkpoint_manager_test.py',
    'orbax/checkpoint/experimental/emergency/multihost_test.py',
    'orbax/checkpoint/experimental/emergency/multi_tier_checkpointing/replicator_checkpoint_manager_test.py',
    'orbax/checkpoint/_src/testing/multiprocess_test.py',
    'orbax/checkpoint/_src/testing/oss/multiprocess_test.py',
]
tagged = yaml.safe_load((checkpoint / 'orbax/checkpoint/_src/testing/oss/tagged_tests_whole_suite.yaml').read_text())
for tag, paths in tagged.items():
  if tag.startswith('processes') and paths:
    extra_ignores.extend(path.replace(':', '/') + '.py' for path in paths)
raise SystemExit(subprocess.call(
    [sys.executable, '-m', 'pytest', '--import-mode=importlib', '-q',
     *['--ignore=' + path for path in extra_ignores]],
    cwd=checkpoint,
))
