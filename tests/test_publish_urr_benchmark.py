import json
from pathlib import Path

import pytest

from tools.publish_urr_benchmark import archive_files, checkpoint_files, digest, verify_archive


def test_archive_preserves_empty_members_and_refuses_mutated_inventory(tmp_path):
    empty = tmp_path/'empty.py'
    empty.write_bytes(b'')
    source = tmp_path/'source.py'
    source.write_bytes(b'original source\n')
    archive = tmp_path/'evidence.tar.gz'
    files = {'package/empty.py':empty,'package/source.py':source}
    inventory = archive_files(files,archive)
    assert inventory['package/empty.py']['size_bytes'] == 0
    verify_archive(archive,inventory)
    other = tmp_path/'same.tar.gz'
    archive_files(files,other)
    assert archive.read_bytes() == other.read_bytes()
    source.write_bytes(b'changed source\n')
    with pytest.raises(ValueError,match='inventory differs'):
        archive_files(files,archive)
    with pytest.raises(ValueError,match='Unsafe'):
        archive_files({'../escaped':empty},tmp_path/'bad.tar.gz')
    assert not (tmp_path/'bad.tar.gz').exists()


def test_only_final_complete_checkpoint_is_selected_without_touching_older(tmp_path):
    old = tmp_path/'checkpoint-000001'
    old.mkdir()
    sentinel = old/'retained'
    sentinel.write_bytes(b'older paid state')
    final = tmp_path/'checkpoint-000002'
    final.mkdir()
    leaf = final/'000000.npy'
    leaf.write_bytes(b'opaque numeric checkpoint leaf')
    manifest = {'schema':'xmimic.training-state/v1','iteration':2,
                'leaves':[{'file':leaf.name,'sha256':digest(leaf),'size':leaf.stat().st_size}]}
    (final/'manifest.json').write_text(json.dumps(manifest))
    receipt = {'completed':True,'checkpoint':str(final),'iterations':2}
    selected = checkpoint_files(tmp_path,receipt)
    assert set(selected) == {'checkpoint-000002/manifest.json','checkpoint-000002/000000.npy'}
    assert sentinel.read_bytes() == b'older paid state'
    leaf.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='leaf bytes changed'):
        checkpoint_files(tmp_path,receipt)
    assert sentinel.read_bytes() == b'older paid state'


def test_checkpoint_receipt_cannot_escape_cell(tmp_path):
    with pytest.raises(ValueError,match='final full checkpoint'):
        checkpoint_files(tmp_path,{'completed':True,'iterations':2,'checkpoint':str(Path('/outside'))})
