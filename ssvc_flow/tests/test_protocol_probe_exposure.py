import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "exposure", Path(__file__).parents[1] / "scripts/protocol_state_probes/audit_prior_exposure.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exposure_keeps_executed_evidence_and_original_train_split(tmp_path):
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({"prompt_id": "u22-id", "raw_text": "[1,2,3,4]"}) + "\n"
    )
    result = module.audit({"u22-id": "scene-id"}, [tmp_path])
    assert result["status"] == "CONTAMINATED_DOWNGRADED"
    assert result["executed_matching_rows"] == 1
    assert result["original_split"] == "train"
    assert result["file_inventory"][0]["sha256"]


def test_missing_evidence_cannot_certify_no_exposure(tmp_path):
    assert module.audit({"u22-id": "scene-id"}, [tmp_path / "absent"])["status"] == "UNVERIFIED"


def test_other_real_outputs_no_match_is_bounded_untouched(tmp_path):
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({"prompt_id": "different-id", "raw_text": "bad"}) + "\n"
    )
    result = module.audit({"u22-id": "scene-id"}, [tmp_path])
    assert result["status"] == "VERIFIED_UNTOUCHED"
    assert result["executed_matching_rows"] == 0
    assert result["files_scanned"] == 1
