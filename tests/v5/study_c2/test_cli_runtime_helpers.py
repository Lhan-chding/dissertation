from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from compensability_v4.qwen.phase5_runtime import phase5_rollout_seed
from compensability_v5.study_c2 import cli, policy_support_runtime


class _FixedParser:
    def __init__(self, namespace: argparse.Namespace) -> None:
        self._namespace = namespace

    def parse_args(self) -> argparse.Namespace:
        return self._namespace


def test_cli_helpers_resolve_fixture_b3_and_legacy_traces(tmp_path: Path) -> None:
    fixture = cli._fixture(28)
    assert fixture == {
        "gpu_invoked": False,
        "schema_version": 2,
        "stage": 28,
        "status": "STUDY_C2_REWARD_NULL_GEOMETRY_AUDIT_FIXTURE_OK",
    }

    with pytest.raises(ValueError, match="--b3-adapter and --b3-sha256"):
        cli._require_b3(argparse.Namespace(b3_adapter=None, b3_sha256=None))

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert cli._require_b3(argparse.Namespace(b3_adapter=adapter, b3_sha256="abc123")) == (
        adapter,
        "abc123",
    )

    explicit = tmp_path / "explicit.jsonl"
    explicit.write_text("{}\n", encoding="utf-8")
    assert cli._legacy_traces(
        argparse.Namespace(trace=[explicit], legacy_root=tmp_path / "unused")
    ) == (explicit,)

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    keep = legacy_root / "raw_trace.jsonl"
    keep.write_text("{}\n", encoding="utf-8")
    ignore = legacy_root / "summary.jsonl"
    ignore.write_text("{}\n", encoding="utf-8")
    target = legacy_root / "raw_link.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    (legacy_root / "raw_symlink.jsonl").symlink_to(target)
    assert cli._legacy_traces(argparse.Namespace(trace=[], legacy_root=legacy_root)) == (
        target,
        keep,
    )

    empty_root = tmp_path / "empty_legacy"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="no raw Study C JSONL traces"):
        cli._legacy_traces(argparse.Namespace(trace=[], legacy_root=empty_root))

    with pytest.raises(ValueError, match="legacy Study C root is unavailable"):
        cli._legacy_traces(
            argparse.Namespace(trace=[], legacy_root=tmp_path / "missing_legacy_root")
        )


def test_cli_not_reached_messages_are_stage_sensitive() -> None:
    with pytest.raises(RuntimeError, match="registered input artifacts"):
        cli._not_reached(21)
    with pytest.raises(RuntimeError, match="frozen-policy support"):
        cli._not_reached(24)


def test_run_registered_covers_fixture_blocked_cpu_gpu_and_placeholder_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "_parser",
        lambda stage: _FixedParser(
            argparse.Namespace(
                fixture_dry_run=True,
                execute=False,
                preflight_only=False,
                config=Path("config.yaml"),
                b3_adapter=None,
                b3_sha256=None,
                ack=None,
                arm=None,
                legacy_root=tmp_path,
                trace=[],
            )
        ),
    )
    assert cli.run_registered(20) == 0
    assert "STUDY_C2_AUDIT_LEGACY_STUDY_C_PARSER_FIXTURE_OK" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "_parser",
        lambda stage: _FixedParser(
            argparse.Namespace(
                fixture_dry_run=False,
                execute=False,
                preflight_only=False,
                config=Path("config.yaml"),
                b3_adapter=None,
                b3_sha256=None,
                ack=None,
                arm=None,
                legacy_root=tmp_path,
                trace=[],
            )
        ),
    )
    assert cli.run_registered(28) == 2
    assert "BLOCKED" in capsys.readouterr().out

    trace = tmp_path / "raw_trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "audit_legacy_trace", lambda **kwargs: {"status": "LEGACY_OK"})
    monkeypatch.setattr(cli, "print_json", lambda payload: print(payload["status"]))
    monkeypatch.setattr(
        cli,
        "_parser",
        lambda stage: _FixedParser(
            argparse.Namespace(
                fixture_dry_run=False,
                execute=True,
                preflight_only=False,
                config=Path("config.yaml"),
                b3_adapter=None,
                b3_sha256=None,
                ack=None,
                arm=None,
                legacy_root=tmp_path,
                trace=[trace],
            )
        ),
    )
    assert cli.run_registered(20) == 0
    assert "LEGACY_OK" in capsys.readouterr().out

    monkeypatch.setattr(cli, "preflight_support", lambda **kwargs: {"status": "PREFLIGHT_OK"})
    monkeypatch.setattr(
        cli,
        "_parser",
        lambda stage: _FixedParser(
            argparse.Namespace(
                fixture_dry_run=False,
                execute=False,
                preflight_only=True,
                config=Path("config.yaml"),
                b3_adapter=tmp_path / "adapter",
                b3_sha256="digest",
                ack=None,
                arm=None,
                legacy_root=tmp_path,
                trace=[],
            )
        ),
    )
    assert cli.run_registered(23) == 0
    assert "PREFLIGHT_OK" in capsys.readouterr().out

    monkeypatch.setattr(
        cli, "run_frozen_policy_support", lambda **kwargs: {"status": "SUPPORT_COMPLETE"}
    )
    monkeypatch.setattr(
        cli,
        "_parser",
        lambda stage: _FixedParser(
            argparse.Namespace(
                fixture_dry_run=False,
                execute=True,
                preflight_only=False,
                config=Path("config.yaml"),
                b3_adapter=tmp_path / "adapter",
                b3_sha256="digest",
                ack="ack",
                arm=None,
                legacy_root=tmp_path,
                trace=[],
            )
        ),
    )
    assert cli.run_registered(23) == 0
    assert "SUPPORT_COMPLETE" in capsys.readouterr().out

    fiber_rows = tmp_path / "reward_fibers.jsonl"
    fiber_rows.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "FIBER_ROWS", fiber_rows)
    monkeypatch.setattr(cli, "SUPPORT_MANIFEST", tmp_path / "support_manifest.json")
    monkeypatch.setattr(
        cli,
        "_parser",
        lambda stage: _FixedParser(
            argparse.Namespace(
                fixture_dry_run=False,
                execute=True,
                preflight_only=False,
                config=Path("config.yaml"),
                b3_adapter=None,
                b3_sha256=None,
                ack=None,
                arm=None,
                legacy_root=tmp_path,
                trace=[],
            )
        ),
    )
    assert cli.run_registered(24) == 2
    assert "run the first Study C2 server boundary" in capsys.readouterr().out

    with pytest.raises(ValueError, match="unregistered Study C2 stage"):
        cli.run_registered(99)


def test_policy_runtime_helpers_validate_tokenizer_and_partial_rollouts() -> None:
    plain = object()
    assert policy_support_runtime._tokenizer(plain) is plain

    class Processor:
        def __init__(self) -> None:
            self.tokenizer = object()

    processor = Processor()
    assert policy_support_runtime._tokenizer(processor) is processor.tokenizer

    class Tokenizer:
        eos_token_id = 99

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "\n"
            assert add_special_tokens is False
            return [13]

    assert policy_support_runtime._newline_eos_ids(Tokenizer()) == [99, 13]

    class MultiTokenTokenizer:
        eos_token_id = 1

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [1, 2]

    with pytest.raises(RuntimeError, match="newline is not one tokenizer token"):
        policy_support_runtime._newline_eos_ids(MultiTokenTokenizer())

    class NoEosTokenizer:
        eos_token_id = None

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [5]

    with pytest.raises(RuntimeError, match="integer EOS token"):
        policy_support_runtime._newline_eos_ids(NoEosTokenizer())

    prompts = (
        {"scene_id": "scene-a", "pair_id": "pair-a"},
        {"scene_id": "scene-b", "pair_id": "pair-b"},
    )
    completed = [
        {
            "scene_id": "scene-a",
            "pair_id": "pair-a",
            "rollout_index": 0,
            "seed": phase5_rollout_seed(17, "scene-a", 0),
            "kind": "X",
        },
        {
            "scene_id": "scene-a",
            "pair_id": "pair-a",
            "rollout_index": 1,
            "seed": phase5_rollout_seed(17, "scene-a", 1),
            "kind": "S",
        },
    ]
    policy_support_runtime._validate_partial_rows(
        completed=completed, prompts=prompts, rollouts_per_prompt=2, seed=17
    )

    with pytest.raises(RuntimeError, match="exceeds registered rollout count"):
        policy_support_runtime._validate_partial_rows(
            completed=completed * 3, prompts=prompts, rollouts_per_prompt=2, seed=17
        )

    drifted = [dict(completed[0]), dict(completed[1], kind="BAD")]
    with pytest.raises(RuntimeError, match="drifted at rollout 1"):
        policy_support_runtime._validate_partial_rows(
            completed=drifted, prompts=prompts, rollouts_per_prompt=2, seed=17
        )
