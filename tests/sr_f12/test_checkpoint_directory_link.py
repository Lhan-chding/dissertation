"""Storage-only compatibility; production code and scientific protocol unchanged."""

from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_training import test_resume_restores_actual_adam_scheduler_cursor_parameters_and_rng as restore_check
from test_endpoint import completed_matrix, audit
from sr_f12.endpoint import bounded_file
from sr_f12.protocol import scientific_matrix


def test_directory_link_preserves_complete_restore(tmp_path):
    target = tmp_path / 'hdd'
    target.mkdir()
    link = tmp_path / 'checkpoints'
    link.symlink_to(target, target_is_directory=True)
    restore_check(link)


def test_eight_future_checkpoint_links_preserve_endpoint_validation(tmp_path):
    root = completed_matrix(tmp_path/'run')
    external = tmp_path/'hdd'
    external.mkdir()
    for row in scientific_matrix():
        if row['seed'] in (71002,71003):
            source = root/'runs'/row['model_id']/'checkpoints'
            target = external/row['model_id']
            shutil.move(str(source),str(target))
            source.symlink_to(target, target_is_directory=True)
    assert audit(root)['status']=='ALL_TWELVE_MAIN_PATHS_VERIFIED'
    link = root/'runs'/scientific_matrix()[-1]['model_id']/'checkpoints'
    outside = tmp_path/'outside.pt'
    outside.write_bytes(b'outside')
    (link/'escape.pt').symlink_to(outside)
    with pytest.raises(PermissionError):
        bounded_file(link,'escape.pt')
