# get_param must not mix shards from different repos, see issue #9.
# Runs offline: the HuggingFace calls are replaced by a fake hub that writes
# real safetensors files, so no network and no model downloads.
# Usage: python3 get_param_test.py   (also collectable by pytest)

import os, shutil, tempfile, torch
from safetensors.torch import save_file
import transformer_tricks as tt

# Model A is sharded, model B is a single file. The filenames deliberately do
# not overlap: that is what makes the old bug silent rather than a clobber.
REPOS = {
  'org/model-a': {
    'model-00001-of-00002.safetensors': {'a.0': torch.zeros(2)},
    'model-00002-of-00002.safetensors': {'a.1': torch.zeros(2)},
  },
  'org/model-b': {
    'model.safetensors': {'b.0': torch.ones(2)},
  },
}


def fake_hub():
  """stand-ins for repo_exists and snapshot_download, no network"""
  def repo_exists(repo):
    return repo in REPOS

  def snapshot_download(repo_id, allow_patterns=None, local_dir=None, **kw):
    os.makedirs(local_dir, exist_ok=True)
    for name, tensors in REPOS[repo_id].items():
      save_file(tensors, os.path.join(local_dir, name), metadata={'repo': repo_id})
    return local_dir

  return repo_exists, snapshot_download


def with_fake_hub(fn):
  """run fn(tmp_dir) against the fake hub, then restore the real one"""
  real_exists, real_download = tt.repo_exists, tt.snapshot_download
  tt.repo_exists, tt.snapshot_download = fake_hub()
  tmp = tempfile.mkdtemp(prefix='get_param_test_')
  try:
    return fn(tmp)
  finally:
    tt.repo_exists, tt.snapshot_download = real_exists, real_download
    shutil.rmtree(tmp, ignore_errors=True)


def test_second_repo_is_not_contaminated_by_the_first():
  """The reported bug: two get_param calls in one process.

  With a single shared download dir, model A's two shards are still sitting
  there when model B is fetched, and the glob picks up all three files. B's
  param dict silently gains A's tensors.
  """
  def body(tmp):
    a = tt.get_param('org/model-a', tmp_dir=tmp)
    b = tt.get_param('org/model-b', tmp_dir=tmp)
    return set(a), set(b)

  keys_a, keys_b = with_fake_hub(body)
  assert keys_a == {'a.0', 'a.1'}, keys_a
  assert keys_b == {'b.0'}, f'model B inherited tensors from model A: {keys_b}'


def test_metadata_comes_from_the_requested_repo():
  """get_meta reads files[0]. With a shared directory that file is whichever
  shard the glob happened to return first, which may belong to another model,
  so the answer was arbitrary rather than reliably wrong. Per-repo dirs plus
  the sorted() make it deterministic and correct."""
  def body(tmp):
    tt.get_param('org/model-a', tmp_dir=tmp)
    _, meta = tt.get_param('org/model-b', get_meta=True, tmp_dir=tmp)
    return meta

  meta = with_fake_hub(body)
  assert meta['repo'] == 'org/model-b', meta


def test_repeated_call_for_the_same_repo_is_stable():
  """Re-fetching one repo must stay identical, so the per-repo directory is
  still reused rather than accumulating duplicates."""
  def body(tmp):
    first = set(tt.get_param('org/model-a', tmp_dir=tmp))
    second = set(tt.get_param('org/model-a', tmp_dir=tmp))
    return first, second

  first, second = with_fake_hub(body)
  assert first == second == {'a.0', 'a.1'}, (first, second)


def test_local_dir_is_read_directly():
  """A path that is not a HuggingFace repo is still treated as a local dir."""
  def body(tmp):
    local = os.path.join(tmp, 'local_model')
    os.makedirs(local)
    save_file({'local.0': torch.zeros(2)}, os.path.join(local, 'model.safetensors'))
    return set(tt.get_param(local, tmp_dir=tmp))

  assert with_fake_hub(body) == {'local.0'}


def test_empty_dir_raises_instead_of_returning_nothing():
  """Previously an empty dir returned {} for get_meta=False and raised
  IndexError on files[0] for get_meta=True. Both are worse than saying so."""
  def body(tmp):
    empty = os.path.join(tmp, 'empty')
    os.makedirs(empty)
    try:
      tt.get_param(empty, tmp_dir=tmp)
    except FileNotFoundError as e:
      return str(e)
    return None

  msg = with_fake_hub(body)
  assert msg is not None, 'expected FileNotFoundError for a dir with no safetensors'
  assert 'safetensors' in msg, msg


if __name__ == '__main__':
  fails = 0
  for name, fn in sorted(globals().items()):
    if name.startswith('test_') and callable(fn):
      try:
        fn()
        print(f'PASS  {name}')
      except AssertionError as e:
        fails += 1
        print(f'FAIL  {name}: {e}')
  print(f'\n{fails} failure(s)')
  raise SystemExit(1 if fails else 0)
