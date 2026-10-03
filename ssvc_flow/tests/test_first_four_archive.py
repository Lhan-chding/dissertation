import gzip,importlib.util,io,tarfile
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('raw_verify',Path(__file__).parents[1]/'scripts/prospective_selection/verify_first_four_archive.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

@pytest.mark.parametrize('names',[['../outside.json'],['same.json','same.json']])
def test_unsafe_or_duplicate_archive_names_rejected(tmp_path,names):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for name in names:
            info=tarfile.TarInfo(name);info.size=2;t.addfile(info,io.BytesIO(b'{}'))
    with pytest.raises(AssertionError):m.main(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()
