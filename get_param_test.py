# get_param must not mix shards from different repos, see issue #9.
# Offline: the two HuggingFace calls are swapped for a fake hub that writes real
# safetensors, so this needs no network and downloads nothing.
# Usage: python3 get_param_test.py   (also collectable by pytest)

import atexit, os, shutil, tempfile, torch
from safetensors.torch import save_file
import transformer_tricks as tt

# A is sharded, B is a single file. The filenames deliberately do not overlap,
# which is what made the old bug silent instead of an obvious clobber.
REPOS = {'org/a': {'model-00001-of-00002.safetensors': {'a.0': torch.zeros(2)},
                   'model-00002-of-00002.safetensors': {'a.1': torch.zeros(2)}},
         'org/b': {'model.safetensors': {'b.0': torch.ones(2)}}}


def fake_download(repo_id, allow_patterns=None, local_dir=None, **kw):
  os.makedirs(local_dir, exist_ok=True)
  for name, tensors in REPOS[repo_id].items():
    save_file(tensors, os.path.join(local_dir, name), metadata={'repo': repo_id})


tt.repo_exists, tt.snapshot_download = lambda r: r in REPOS, fake_download
TMP = tempfile.mkdtemp(prefix='get_param_test_')
atexit.register(shutil.rmtree, TMP, True)


def test_second_repo_is_not_contaminated():
  """The reported bug. Sharing one download dir leaves A's two shards in place
  when B arrives, so the glob returns all three and B silently gains A's."""
  assert set(tt.get_param('org/a', tmp_dir=TMP)) == {'a.0', 'a.1'}
  assert set(tt.get_param('org/b', tmp_dir=TMP)) == {'b.0'}
  # refetching A must be stable, so the per-repo dir is reused, not accumulated
  assert set(tt.get_param('org/a', tmp_dir=TMP)) == {'a.0', 'a.1'}


def test_metadata_comes_from_the_requested_repo():
  """get_meta reads files[0], which under a shared dir was whichever shard the
  glob happened to return first, so the answer was arbitrary rather than
  reliably wrong. Fetching A first is what makes this test mean anything: without
  it the directory is clean either way and the assert passes even unfixed."""
  tt.get_param('org/a', tmp_dir=TMP)
  _, meta = tt.get_param('org/b', get_meta=True, tmp_dir=TMP)
  assert meta['repo'] == 'org/b', meta


def test_empty_dir_raises():
  """Was {} for get_meta=False and IndexError on files[0] for get_meta=True.
  Both are worse than saying so."""
  empty = os.path.join(TMP, 'empty')
  os.makedirs(empty, exist_ok=True)
  try:
    tt.get_param(empty, tmp_dir=TMP)
  except FileNotFoundError as e:
    assert 'safetensors' in str(e), e
  else:
    assert False, 'expected FileNotFoundError for a dir with no safetensors'


if __name__ == '__main__':
  fails = 0
  for name, fn in sorted(globals().items()):
    if name.startswith('test_'):
      try:
        fn()
        print(f'PASS  {name}')
      except AssertionError as e:
        fails += 1
        print(f'FAIL  {name}: {e}')
  raise SystemExit(1 if fails else 0)
