from pathlib import Path

from src.audit_r0_remaining import (
    audit_cross_split,
    audit_symbolic_prompts,
    compare_contact_manifest,
    decode_raw_completion,
    select_36,
)


class Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            {1: "[", 2: "1", 3: ",", 4: "2", 5: "]", 9: "<think>"}.get(i, "?") for i in ids
        )


def test_decode_only_trailing_eos_preserves_think():
    text, ids = decode_raw_completion([9, 1, 2, 3, 4, 5, 248046], Tok(), [248044, 248046])
    assert text == "<think>[1,2]"
    assert ids[-1] == 5


def test_cross_split_passes_generated_dataset():
    r = audit_cross_split(Path("data/generated"))
    assert r["status"] == "PASS"
    assert not r["base_scene_collisions"]


def test_hash_selection_and_reuse_manifest():
    root = Path("data/generated")
    selected, missing = select_36(root)
    assert len(selected) == 36 and not missing
    cm = Path("docs/local_evidence/P0/contact_sheets_manifest.json")
    r = compare_contact_manifest(root, cm)
    assert r["can_reuse_human_review"] is True


def test_symbolic_prompt_no_undeclared_labels():
    r = audit_symbolic_prompts(Path("data/generated"))
    assert r["status"] == "PASS"
    assert r["checked"] == 144
