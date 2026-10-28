"""Complete CPU-only orchestration fixture; no real model or scientific acceptance.

The loader and registration verifier are explicitly mocked. All raw trajectories
and mock runtime/accounting receipts are marked SYNTHETIC_TEST. This exercises
production orchestration/scoring/alignment mechanics, never actual GPU evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from sr_f4.common import append
from sr_f6 import confirm_census as census
from sr_f6 import confirm_registration, confirm_runtime
from sr_f6.common import digest, file_hash, read, write_once
from sr_f6.interface_v3_protocol import VERSION, public_prompt
from sr_f6.runtime import PROTOCOL_VERSION, generation_recipe
from srf6_ref.design import build_bank
from srf6_ref.measurement import full_truth_response


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def frozen_inputs():
    roots = build_bank(census.SPLIT, 8, census.MASTER)
    tasks = sorted(
        (task for root in roots for views in root["tasks"].values() for task in views),
        key=lambda task: task["qid"],
    )
    public = []
    for task in tasks:
        prompt = public_prompt(task["world"]["categories"], task["query"])
        public.append(
            dict(
                **{
                    key: task[key]
                    for key in ("qid", "root_id", "family", "factor", "variant", "split")
                },
                prompt_version=VERSION,
                protocol="structured_v4",
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                model_input=dict(
                    prompt=prompt,
                    image_file=f"DATA/images/{digest(task['world'])}.png",
                    image_sha256="a" * 64,
                ),
            )
        )
    private = {task["qid"]: task for task in tasks}
    parents = {parent: str(index + 1) * 64 for index, parent in enumerate(census.PARENTS)}
    plan = confirm_registration.make_plan(public, private, parents)
    routes = {
        row["qid"]: dict(
            input_tensor_hash=digest(row["model_input"]),
            processor_hash="b" * 64,
            chat_template_hash="c" * 64,
            chat_text_sha256=row["prompt_sha256"],
        )
        for row in public
    }
    return SimpleNamespace(
        public=public, private=private, parents=parents, plan=plan, routes=routes
    )


def test_complete_registered_shard_preserves_every_draw_and_first_alignment(
    frozen_inputs, tmp_path, monkeypatch, capsys
):
    frozen = frozen_inputs
    root = tmp_path.resolve()
    parent, shard, task_id = "b0", 1, "LH-SRF6-Confirm-321-b0-1"
    assignment = next(
        worker
        for worker in frozen.plan["workers"]
        if worker["parent"] == parent and worker["shard"] == shard
    )
    context = dict(
        root=root,
        public=frozen.public,
        private=frozen.private,
        spec=frozen.plan,
        spec_object_sha256=digest(frozen.plan),
        registration_sha256="d" * 64,
        registration=dict(
            source_manifest_sha256="e" * 64,
            data_sha256="f" * 64,
            parent_content_hashes=frozen.parents,
        ),
        preflight=dict(
            inputs=[dict(qid=qid, routing=route) for qid, route in frozen.routes.items()]
        ),
    )
    context_before = copy.deepcopy(context)
    verification, loads, generated_qids, replay_calls, persisted_draws = [], [], [], [], []
    ledger = root / f"CONFIRM/accounting/{task_id}.jsonl"
    tensor_pin = "9" * 64

    def verify(candidate, *, parent, require_preflight):
        assert candidate == root and parent == "b0" and require_preflight is True
        verification.append((candidate, parent, require_preflight))
        return context

    class SyntheticRuntime:
        torch = torch

        def account(self, qid, kind, count, **metadata):
            append(
                ledger,
                dict(
                    execution_origin="SYNTHETIC_TEST",
                    task_id=task_id,
                    parent=parent,
                    kind=kind,
                    count=count,
                    metadata=dict(
                        metadata,
                        qid=qid,
                        phase=census.PHASE,
                        confirmation_version="srf6-confirm-v3",
                        registration_sha256=context["registration_sha256"],
                    ),
                ),
            )

        def policy_tensor_identity(self):
            return tensor_pin

        def generate_registered(self, qid, *, on_completion):
            index = len(generated_qids)
            assert qid == assignment["qids"][index]
            if index:
                assert replay_calls == [assignment["first_qid"]]
                assert (root / f"CONFIRM/alignment/{task_id}.json").is_file()
            generated_qids.append(qid)
            item = next(row for row in frozen.public if row["qid"] == qid)
            task = frozen.private[qid]
            spec = census.group_spec(context, parent, item)
            attempt = read(root / f"CONFIRM/attempts/{parent}/{qid}.json")
            assert attempt["spec"] == spec and attempt["task_id"] == task_id
            raw_path = root / f"CONFIRM/raw/{parent}/{qid}.jsonl"
            group_path = root / f"CONFIRM/groups/{parent}/{qid}.json"
            self.account(
                qid, "completion_attempts", 8, group_seed=spec["seeds"][0], protocol="structured"
            )
            for draw, seed in enumerate(spec["seeds"]):
                row = dict(
                    **{
                        key: spec[key]
                        for key in (
                            "phase",
                            "registration_sha256",
                            "source_sha256",
                            "data_hash",
                            "parent_content_hash",
                        )
                    },
                    execution_origin="SYNTHETIC_TEST",
                    fixture="NO_ACTUAL_MODEL_CALL",
                    policy_tensor_hash=tensor_pin,
                    generation_status="COMPLETE",
                    technical_validation_errors=[],
                    execution_mode="READONLY_SCHEMA_CONFIRM",
                    confirmation_version="srf6-confirm-v3",
                    optimizer_updates=0,
                    step=0,
                    policy_step=0,
                    protocol="structured_v4",
                    prompt_version=VERSION,
                    sr_f6_protocol_version=PROTOCOL_VERSION,
                    seed_semantics="independent_per_draw_torch_generator",
                    group_size=8,
                    group_row_index=draw,
                    group_seed=spec["seeds"][0],
                    seed=seed,
                    adapter_identity=dict(parent=parent, step=0, run_id=task_id),
                    tokens=[10],
                    raw_tokens=[10],
                    sampler_logprobs=[-0.5],
                    completion_token_count=1,
                    max_new_tokens=1536,
                    sampler_logprob_source="actual_generate_raw_logits_selected_token",
                    finish_reason="eos" if draw == 0 else "balanced",
                    full_text=full_truth_response(task["world"], task["query"]),
                    task_id=task_id,
                    run_id=task_id,
                    run_root=str(root),
                    model_input=copy.deepcopy(item["model_input"]),
                    prompt_sha256=item["prompt_sha256"],
                    sampling_parameters=generation_recipe(),
                    prompt_tensor_hash=frozen.routes[qid]["input_tensor_hash"],
                    input_routing=copy.deepcopy(frozen.routes[qid]),
                )
                self.account(
                    qid, "generated_tokens", 1, group_seed=spec["seeds"][0], row_index=draw
                )
                on_completion(row)
                # Callback is synchronous: each draw is on disk before the next.
                saved = _rows(raw_path)
                assert len(saved) == draw + 1 and saved[-1] == row
                assert not group_path.exists()
                persisted_draws.append((qid, draw))

        def replay_alignment(self, rows, *, on_replay):
            qid = assignment["first_qid"]
            assert not replay_calls and generated_qids == [qid]
            assert rows == _rows(root / f"CONFIRM/raw/{parent}/{qid}.jsonl")
            assert [row["group_row_index"] for row in rows] == list(range(8))
            assert rows[0]["finish_reason"] == "eos" and "score" not in rows[0]
            replay_calls.append(qid)
            output = []
            for draw, row in enumerate(rows):
                self.account(
                    qid,
                    "extra_forward_sequences",
                    1,
                    purpose="confirm_token_alignment",
                    completion_tokens=len(row["tokens"]),
                )
                value = torch.tensor(row["sampler_logprobs"], dtype=torch.float32)
                on_replay(draw, value.clone())
                output.append(value)
            return output

    runtime = SyntheticRuntime()

    def load(given, chosen_parent, given_task):
        assert given is context and chosen_parent == parent and given_task == task_id
        loads.append(given_task)
        slot = read(root / f"CONFIRM/execution_slots/{parent}-{shard}.json")
        assert slot == read(root / f"CONFIRM/executions/{task_id}.json")
        write_once(
            root / f"CONFIRM/runtime_identity/{task_id}.json",
            dict(execution_origin="SYNTHETIC_TEST", task_id=task_id, phase=census.PHASE),
        )
        return runtime

    monkeypatch.setattr(confirm_registration, "verify_registered", verify)
    monkeypatch.setattr(confirm_runtime, "load_confirm", load)
    receipt = census.run(root, parent, shard, task_id)
    assert generated_qids == assignment["qids"] and len(generated_qids) == 192
    assert persisted_draws == [(qid, draw) for qid in assignment["qids"] for draw in range(8)]
    assert replay_calls == [assignment["first_qid"]] and loads == [task_id]
    assert verification == [(root, parent, True)] * 2
    assert context == context_before
    assert receipt == read(root / f"CONFIRM/workers/{task_id}.json")
    assert receipt["responses"] == receipt["generated_this_process"] == 1536
    assert receipt["actual_generated_responses_this_process"] == 1536
    assert (
        receipt["optimizer_updates"] == receipt["recovered_raw_rows"] == receipt["reused_rows"] == 0
    )
    assert receipt["training_allowed"] is False and receipt["gate_status"] == "NOT_SIGNED"
    assert receipt["frozen_spec_sha256"] == digest(frozen.plan)
    alignment_path = root / f"CONFIRM/alignment/{task_id}.json"
    alignment = read(alignment_path)
    assert receipt["generation_alignment_sha256"] == file_hash(alignment_path)
    assert alignment["checked_sequences"] == len(alignment["traces"]) == 8
    assert alignment["new_generated_responses"] == alignment["optimizer_updates"] == 0
    assert alignment["diagnostics"]["sampler_train_max_absolute_difference"] == 0
    assert len(list((root / "CONFIRM/groups/b0").glob("*.json"))) == 192
    assert len(list((root / "CONFIRM/attempts/b0").glob("*.json"))) == 192
    originals = [_rows(root / f"CONFIRM/raw/{parent}/{qid}.jsonl") for qid in generated_qids]
    assert sum(map(len, originals)) == 1536
    assert all(row["execution_origin"] == "SYNTHETIC_TEST" for group in originals for row in group)
    first = read(root / f"CONFIRM/groups/{parent}/{assignment['first_qid']}.json")
    assert first["records"][0]["finish_reason"] == first["records"][0]["raw_finish_reason"] == "eos"
    assert first["records"][0]["finished"] == 0
    assert set(first["credit"]) == {"J", "COND", "GLOBAL", "SHUF"}
    events = _rows(ledger)
    assert Counter(event["kind"] for event in events) == dict(
        completion_attempts=192, generated_tokens=1536, extra_forward_sequences=8
    )
    assert sum(event["count"] for event in events if event["kind"] == "completion_attempts") == 1536
    assert not list((root / "CONFIRM/failures").glob("*.json"))
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line)["qid"] for line in lines] == assignment["qids"]
    with pytest.raises(PermissionError, match="no automatic retry"):
        census.run(root, parent, shard, "LH-SRF6-Confirm-322-b0-1")
    assert loads == [task_id] and len(persisted_draws) == 1536
    assert read(root / f"CONFIRM/workers/{task_id}.json") == receipt
    assert not list((root / "CONFIRM/failures").glob("*.json"))
