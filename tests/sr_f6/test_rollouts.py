"""Fake-runtime CPU persistence tests, never actual model/GPU evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from mm_core.training import capture_rng, state_hash, trainable_state
from mm_core.vl_runtime import hash_json
from sr_f3_ce.training import fresh_optimizer
from sr_f6 import rollouts
from sr_f6.common import digest
from sr_f6.data import public_prompt
from sr_f6.planning import draw_seed, root_schedule
from sr_f6.transactions import RunTransactions
from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.design import make_root
from srf6_ref.measurement import full_truth_response


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(v) + "\n" for v in value) if lines else json.dumps(value)
    path.write_text(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_sampler():
    return dict(
        next_step=1,
        committed_log_step=0,
        consumed_qids={},
        logical_calls=0,
        actual_calls=0,
        rollout_cursor=dict(step=0, groups=0, rows=0),
        rollout_rng_reference=None,
    )


class FakeRuntime:
    """Emits fixture text, tokens and identities while advancing all CPU RNGs."""

    def __init__(self, model, identity, text):
        self.model, self.identity, self.text = model, identity, text
        self.calls = 0
        self.invocations = 0
        self.received = []
        self.defect = None
        self.finish_override = None
        self.audit_hook = None

    def policy_tensor_identity(self):
        return self._srf6_policy_identity[1]

    def generate_group(self, safe, root, seeds, *, protocol, on_completion):
        assert set(safe) == {"prompt", "image_file", "image_sha256"}
        assert protocol == "structured_v4" and len(seeds) == 8
        self.invocations += 1
        self.received.append(copy.deepcopy(safe))
        if self.audit_hook:
            self.audit_hook("before_generation", self.invocations - 1, 0)
        if self.defect == "unknown_call":
            raise RuntimeError("fixture call interrupted before any callback")
        output = []
        recipe = dict(max_new_tokens=1536, temperature=1.0, top_p=1.0, top_k=0, do_sample=True)
        for draw, seed in enumerate(seeds):
            self.calls += 1
            tokens = [
                random.randrange(1, 100),
                int(np.random.randint(1, 100)),
                int(torch.randint(1, 100, ()).item()),
            ]
            row = dict(
                seed=seed,
                group_seed=seeds[0],
                group_size=8,
                group_row_index=draw,
                model_input=copy.deepcopy(safe),
                run_root=str(root),
                run_id=self.identity["run_id"],
                source_sha256=self.identity["source_sha256"],
                data_hash=self.identity["data_hash"],
                phase=self.identity["phase"],
                execution_mode="TRAINING",
                policy_hash=self.policy_tensor_identity(),
                policy_tensor_hash=self.policy_tensor_identity(),
                parent_content_hash="f" * 64,
                policy_step=self.identity["step"],
                step=self.identity["step"] + 1,
                adapter_identity=dict(
                    run_id=self.identity["run_id"],
                    parent=self.identity["parent_id"],
                    step=self.identity["step"],
                ),
                tokens=tokens,
                raw_tokens=list(tokens),
                completion_token_count=len(tokens),
                sampler_logprobs=[-1.0, -2.0, -3.0],
                sampler_logprob_source="actual_generate_raw_logits_selected_token",
                sampling_parameters=recipe,
                sampling_hash=hash_json(recipe),
                max_new_tokens=1536,
                protocol=protocol,
                native_prompt_inputs={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]},
                prompt_tensor_hash="d" * 64,
                prompt_sha256=hashlib.sha256(safe["prompt"].encode()).hexdigest(),
                full_text=self.text if seed % 3 else "invalid fixture response",
                finish_reason="balanced",
                generation_status="COMPLETE",
                technical_validation_errors=[],
            )
            if self.finish_override and draw == 0:
                row["finish_reason"] = self.finish_override
                if self.finish_override == "length":
                    row.update(
                        tokens=[1] * 1536,
                        raw_tokens=[1] * 1536,
                        completion_token_count=1536,
                        sampler_logprobs=[-1.0] * 1536,
                    )
            if self.defect == "technical_invalid" and draw == 0:
                row.update(
                    generation_status="TECHNICAL_INVALID",
                    technical_validation_errors=["FIXTURE_INVALID"],
                )
            elif self.defect == "nonfinite" and draw == 0:
                row.update(
                    generation_status="TECHNICAL_INVALID",
                    technical_validation_errors=["NONFINITE_SAMPLER_LOGPROB"],
                    sampler_logprobs=[float("nan")] * 3,
                )
            elif self.defect == "wrong_source" and draw == 0:
                row["source_sha256"] = "0" * 64
            elif self.defect == "wrong_seed" and draw == 0:
                row["seed"] += 1
            elif self.defect == "duplicate_seed" and draw == 1:
                row["seed"] = seeds[0]
            elif self.defect == "over_tokens" and draw == 0:
                row.update(
                    tokens=[1] * 1537,
                    raw_tokens=[1] * 1537,
                    completion_token_count=1537,
                    sampler_logprobs=[-1.0] * 1537,
                )
            elif self.defect == "mutate_parameters" and draw == 0:
                with torch.no_grad():
                    next(self.model.parameters()).add_(1)
            on_completion(row)
            if self.audit_hook:
                self.audit_hook("after_callback", self.invocations - 1, draw + 1)
            output.append(row)
        if self.defect == "return_mismatch":
            output = copy.deepcopy(output)
            output[0]["tokens"][0] += 1
        return output


@pytest.fixture(scope="module")
def prototype():
    return make_root("DEV_PILOT", "BINDING", 0, 2026102712)["tasks"]["P1"][0]


def prepare(base, prototype, *, formal=False, edit_registration=None, phase_override=None):
    random.seed(807)
    np.random.seed(807)
    torch.manual_seed(807)
    experiment, run = base / "experiment", base / "run"
    experiment.mkdir(parents=True)
    run.mkdir()
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    config_hash = write(experiment / "CONFIG.json", config)
    split, count, steps, arm = ("TRAIN", 48, 72, "U00") if formal else ("DEV_PILOT", 8, 12, "U11")
    schedule = root_schedule(count, config["master_seed"], "b0", 0, split=split)
    public, private = [], []
    image_file = "DATA/images/fixture.png"
    image = experiment / image_file
    image.parent.mkdir(parents=True)
    image.write_bytes(b"FAKE_RUNTIME_IMAGE_NOT_PROCESSOR_EVIDENCE")
    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    for family in FAMILIES:
        for index in range(count):
            rid = f"{split}_{family}_{index:04d}"
            for factor in ("P0", "P1"):
                for variant in range(4):
                    task = {
                        **copy.deepcopy(prototype),
                        "qid": f"{rid}_{factor}_v{variant}",
                        "root_id": rid,
                        "family": family,
                        "split": split,
                        "factor": factor,
                        "variant": variant,
                    }
                    private.append(task)
                    item = {
                        key: task[key]
                        for key in ("qid", "root_id", "family", "split", "factor", "variant")
                    }
                    public.append(
                        dict(
                            **item,
                            protocol="structured_v4",
                            model_input=dict(
                                prompt=public_prompt(task),
                                image_file=image_file,
                                image_sha256=image_hash,
                            ),
                        )
                    )
    entries, inputs = [], {}
    for name, relative, value, lines in (
        ("public", f"DATA/{split}/MODEL_INPUTS.jsonl", public, True),
        ("private", f"DATA/{split}/TASKS_PRIVATE.jsonl", private, True),
        ("schedule", f"DATA/SCHEDULES/{split}-b0-r0.json", schedule, False),
    ):
        content_hash = write(experiment / relative, value, lines=lines)
        inputs[name] = dict(path=relative, sha256=content_hash)
        entries.append(
            dict(path=relative, sha256=content_hash, bytes=(experiment / relative).stat().st_size)
        )
    entries.append(dict(path=image_file, sha256=image_hash, bytes=image.stat().st_size))
    data_hash = write(
        experiment / "DATA/MANIFEST.json", dict(config_sha256=digest(config), files=entries)
    )
    model = torch.nn.Linear(2, 1).float()
    optimizer, scheduler = fresh_optimizer(model)
    identity = dict(
        experiment_id=config["experiment_id"],
        run_id=f"fixture/{split}/b0/r0/{arm}",
        phase="FORMAL_V1" if formal else "PILOT_V1",
        parent="b0",
        arm=arm,
        repeat=0,
        lane="MAIN",
        max_steps=steps,
        source_sha256="a" * 64,
        data_sha256=data_hash,
        config_sha256=config_hash,
        reference_sha256=state_hash(trainable_state(model)),
    )
    if phase_override is not None:
        identity["phase"] = phase_override
    domain = "formal_shuffle" if formal else "pilot_shuffle"
    registration = dict(
        format=rollouts.FORMAT,
        identity=dict(identity),
        split=split,
        factor="P0" if formal else "P1",
        master_seed=config["master_seed"],
        schedule=schedule,
        inputs=inputs,
        shuffle_domain=domain,
        draw_seeds=[
            [
                [
                    draw_seed(config["master_seed"], "b0", 0, step, group // 4, group % 4, draw)
                    for draw in range(8)
                ]
                for group in range(16)
            ]
            for step in range(1, steps + 1)
        ],
        shuffle_seeds=[
            [
                seed_for(config["master_seed"], domain, "b0", 0, arm, step, group)
                for group in range(16)
            ]
            for step in range(1, steps + 1)
        ],
    )
    if edit_registration:
        edit_registration(registration)
    ref = dict(
        path="ROLLOUT_REGISTRATION.json",
        sha256=write(run / "ROLLOUT_REGISTRATION.json", registration),
    )
    identity["rollout_registration"] = ref
    runtime = FakeRuntime(
        model,
        dict(
            run_id=identity["run_id"],
            parent_id="b0",
            reference_parent="b0",
            step=0,
            source_sha256=identity["source_sha256"],
            data_hash=data_hash,
            reference_hash=identity["reference_sha256"],
            phase=identity["phase"],
            execution_mode="TRAINING",
            run_root=str(experiment),
        ),
        full_truth_response(prototype["world"], prototype["query"]),
    )
    return dict(
        experiment=experiment,
        run=run,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        identity=identity,
        ref=ref,
    )


def collect(tx, case, **kwargs):
    return rollouts.collect_step(
        tx,
        case["runtime"],
        experiment_root=case["experiment"],
        registration_ref=case["ref"],
        **kwargs,
    )


def start(tx, case):
    return tx.commit(
        case["runtime"],
        case["optimizer"],
        case["scheduler"],
        step=0,
        sampler_state=initial_sampler(),
    )


def test_full_original_rollout_and_credit_match_transaction_schema(tmp_path, prototype):
    case = prepare(tmp_path, prototype)
    runtime = case["runtime"]
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        before = state_hash(trainable_state(runtime.model))
        result = collect(tx, case)
        after_rng = state_hash(capture_rng())
        assert len(result["records"]) == 128 and len(result["credits"]) == 16
        assert len({r["root_id"] for r in result["records"]}) == 4
        assert runtime.calls == result["new_model_calls"] == 128
        assert state_hash(trainable_state(runtime.model)) == before
        assert runtime.identity["step"] == 0
        assert result["sampler_state"]["rollout_rng_reference"] == result["references"]["rollout"]
        assert result["gate_status"] == "NOT_SIGNED" and result["training_authorized"] is False
        assert all(
            r["policy_hash"] == before and r["policy_hash"] != r["parent_content_hash"]
            for r in result["records"]
        )
        assert any(r["J"] == 0 for r in result["records"])
        assert any(r["S"] is None for r in result["records"])
        random.random()
        np.random.random()
        torch.rand(1)
        again = collect(tx, case)
        assert again["records"] == result["records"] and again["credits"] == result["credits"]
        assert again["new_model_calls"] == 0 and again["reused_complete_groups"] == 16
        assert runtime.calls == 128 and state_hash(capture_rng()) == after_rng
        assert len(tx.published()) == 1


@pytest.mark.parametrize(
    "location,group",
    [("after_complete_g8", 0), ("after_complete_g8", 7), ("before_score", 16), ("after_ready", 16)],
)
def test_complete_g8_recovery_preserves_tokens_credit_and_rng(tmp_path, prototype, location, group):
    baseline = prepare(tmp_path / "baseline", prototype)
    with RunTransactions(baseline["run"], baseline["identity"]) as tx:
        start(tx, baseline)
        expected = collect(tx, baseline)
        expected_rng = state_hash(capture_rng())
    case = prepare(tmp_path / "crashed", prototype)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)

        def fail(name, index):
            if name == location and index == group:
                raise RuntimeError("fixture interruption")

        with pytest.raises(RuntimeError, match="fixture interruption"):
            collect(tx, case, failpoint=fail)
        calls = case["runtime"].calls
        tx.restore_latest(case["runtime"], case["optimizer"], case["scheduler"])
        actual = collect(tx, case)
        assert case["runtime"].calls == 128
        assert actual["new_model_calls"] == 128 - calls
        assert state_hash(capture_rng()) == expected_rng
        for first, second in zip(actual["records"], expected["records"], strict=True):
            for key in (
                "tokens",
                "sampler_logprobs",
                "native_prompt_inputs",
                "J",
                "S",
                "B",
                "row_id",
                "token_path_sha256",
            ):
                assert first[key] == second[key]
        assert actual["credits"] == expected["credits"]
        assert (case["run"] / "rollouts/step-001/INCIDENTS.jsonl").exists()


@pytest.mark.parametrize("draws", [1, 3, 8])
def test_partial_or_eight_raw_without_complete_marker_never_resamples(tmp_path, prototype, draws):
    case = prepare(tmp_path, prototype)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        emitted = 0

        def fail(name, group):
            nonlocal emitted
            if name == "after_raw_draw":
                emitted += 1
                if emitted == draws:
                    raise RuntimeError("interrupted callback")

        with pytest.raises(RuntimeError, match="interrupted callback"):
            collect(tx, case, failpoint=fail)
        raw = case["run"] / "rollouts/step-001/groups/00/raw.jsonl"
        assert len(raw.read_text().splitlines()) == draws
        assert not raw.with_name("COMPLETE.json").exists()
        before = raw.read_bytes()
        tx.restore_latest(case["runtime"], case["optimizer"], case["scheduler"])
        with pytest.raises(rollouts.IncompleteRollout, match="PARTIAL_OR_UNKNOWN_CALL"):
            collect(tx, case)
        assert case["runtime"].calls == draws and raw.read_bytes() == before
        assert case["runtime"].invocations == 1


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_call",
        "technical_invalid",
        "nonfinite",
        "wrong_source",
        "wrong_seed",
        "duplicate_seed",
        "over_tokens",
        "return_mismatch",
        "mutate_parameters",
    ],
)
def test_backend_failure_is_preserved_and_cannot_be_retried(tmp_path, prototype, defect):
    case = prepare(tmp_path, prototype)
    case["runtime"].defect = defect
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        with pytest.raises((RuntimeError, PermissionError)):
            collect(tx, case)
        calls = case["runtime"].calls
        assert (case["run"] / "rollouts/step-001/INCIDENTS.jsonl").exists()
        assert not (case["run"] / "rollouts/step-001/READY.json").exists()
        tx.restore_latest(case["runtime"], case["optimizer"], case["scheduler"])
        case["runtime"].defect = None
        with pytest.raises((rollouts.IncompleteRollout, PermissionError)):
            collect(tx, case)
        assert case["runtime"].calls == calls


@pytest.mark.parametrize(
    "defect", ["test_split", "schedule", "draw", "shuffle", "domain", "wrong_input_split"]
)
def test_frozen_but_unregistered_schedule_seed_or_split_is_rejected(tmp_path, prototype, defect):
    def edit(spec):
        if defect == "test_split":
            spec["split"] = "TEST_ID"
        elif defect == "schedule":
            spec["schedule"][0].reverse()
        elif defect == "draw":
            spec["draw_seeds"][0][0][0] += 1
        elif defect == "shuffle":
            spec["shuffle_seeds"][0][0] += 1
        elif defect == "domain":
            spec["shuffle_domain"] = "redraw_until_changed"
        else:
            spec["inputs"]["public"]["path"] = "DATA/TEST_ID/MODEL_INPUTS.jsonl"

    case = prepare(tmp_path, prototype, edit_registration=edit)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        with pytest.raises(PermissionError):
            collect(tx, case)
        assert case["runtime"].calls == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("phase", "OTHER"),
        ("execution_mode", "EVALUATION"),
        ("source_sha256", "f" * 64),
        ("data_hash", "f" * 64),
        ("step", 1),
    ],
)
def test_wrong_runtime_identity_is_rejected_before_generation(tmp_path, prototype, key, value):
    case = prepare(tmp_path, prototype)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        case["runtime"].identity[key] = value
        with pytest.raises(PermissionError):
            collect(tx, case)
        assert case["runtime"].calls == 0


def test_lock_step0_and_clean_committed_policy_are_required(tmp_path, prototype):
    case = prepare(tmp_path, prototype)
    tx = RunTransactions(case["run"], case["identity"])
    with pytest.raises(PermissionError, match="lock"):
        collect(tx, case)
    with tx:
        with pytest.raises(PermissionError, match="Step zero"):
            collect(tx, case)
        start(tx, case)
        with torch.no_grad():
            next(case["runtime"].model.parameters()).add_(1)
        with pytest.raises(PermissionError, match="in-memory policy changed"):
            collect(tx, case)
        assert case["runtime"].calls == 0


def test_unchanged_split_files_are_not_rehashed_but_mutation_invalidates_cache(
    tmp_path, prototype, monkeypatch
):
    case = prepare(tmp_path, prototype)
    real_read, reads = rollouts._read_bytes, []

    def counted(path):
        if path.is_relative_to(case["experiment"]):
            reads.append(path)
        return real_read(path)

    monkeypatch.setattr(rollouts, "_read_bytes", counted)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        collect(tx, case)
        count = len(reads)
        collect(tx, case)
        assert len(reads) == count == 5  # CONFIG, manifest and three split inputs only.
        path = case["experiment"] / "DATA/DEV_PILOT/MODEL_INPUTS.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(PermissionError, match="content differs"):
            collect(tx, case)
        assert case["runtime"].calls == 128


def test_formal_warm_cache_does_not_rescan_split_or_repeated_image_paths(
    tmp_path, prototype, monkeypatch
):
    case = prepare(tmp_path, prototype, formal=True)
    real_path, real_prompt = rollouts._path, rollouts.public_prompt
    paths, prompts = [], []

    def counted_path(root, relative):
        if relative.startswith("DATA/images/"):
            paths.append(relative)
        return real_path(root, relative)

    def counted_prompt(task):
        prompts.append(task["qid"])
        return real_prompt(task)

    monkeypatch.setattr(rollouts, "_path", counted_path)
    monkeypatch.setattr(rollouts, "public_prompt", counted_prompt)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        collect(tx, case)
        assert len(prompts) == 2304
        assert paths == ["DATA/images/fixture.png"]
        prompts.clear()
        paths.clear()
        reused = collect(tx, case)
        assert reused["new_model_calls"] == 0
        assert prompts == []  # Frozen whole-split semantic validation is cached.
        assert paths == ["DATA/images/fixture.png"]  # Only this step's unique image.
        image = case["experiment"] / "DATA/images/fixture.png"
        backup = image.with_suffix(".backup")
        image.rename(backup)
        image.symlink_to(backup)
        with pytest.raises(PermissionError, match="Symlink"):
            collect(tx, case)
        assert case["runtime"].calls == 128


@pytest.mark.parametrize("artifact", ["raw", "rng", "ready", "credit"])
def test_complete_cache_tampering_is_not_recollected(tmp_path, prototype, artifact):
    case = prepare(tmp_path, prototype)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        collect(tx, case)
        paths = dict(
            raw="groups/00/raw.jsonl",
            rng="groups/00/rng-after.pt",
            ready="READY.json",
            credit="credit.json",
        )
        path = case["run"] / "rollouts/step-001" / paths[artifact]
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises((PermissionError, ValueError)):
            collect(tx, case)
        assert case["runtime"].calls == 128


def test_formal_uses_train_p0_and_formal_shuffle_domain(tmp_path, prototype):
    case = prepare(tmp_path, prototype, formal=True)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        result = collect(tx, case)
        assert all(row["split"] == "TRAIN" and row["factor"] == "P0" for row in result["records"])
        assert all(
            group["arm"] == "U00" and group["applied"] == [0.0] * 8 for group in result["credits"]
        )


@pytest.mark.parametrize("reason", ["eos", "length"])
def test_incomplete_finish_retains_raw_reason_and_has_no_auxiliary_credit(
    tmp_path, prototype, reason
):
    case = prepare(tmp_path, prototype)
    case["runtime"].finish_override = reason
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        result = collect(tx, case)
        for row, group in zip(result["records"][::8], result["credits"], strict=True):
            assert row["finish_reason"] == row["generation_finish_reason"] == reason
            assert row["score"]["finish_reason"] == "length"
            assert row["finished"] == row["J"] == 0
            assert row["S"] is None and row["B"] is None
            assert group["applied"][0] == 0
        assert len(result["records"]) == 128


@pytest.mark.parametrize(
    "formal,phase", [(True, "PILOT_V1"), (False, "FORMAL_V1"), (False, "OTHER")]
)
def test_phase_cannot_select_another_stage_budget_or_split(tmp_path, prototype, formal, phase):
    case = prepare(tmp_path, prototype, formal=formal, phase_override=phase)
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        with pytest.raises(PermissionError, match=r"[Pp]hase"):
            collect(tx, case)
        assert case["runtime"].calls == 0


def test_started_precedes_generation_and_each_callback_is_already_persisted(tmp_path, prototype):
    case = prepare(tmp_path, prototype)
    seen = []

    def audit(event, group, count):
        directory = case["run"] / "rollouts/step-001/groups" / f"{group:02d}"
        assert (directory / "STARTED.json").is_file()
        if event == "after_callback":
            assert len((directory / "raw.jsonl").read_text().splitlines()) == count
            seen.append((group, count))

    case["runtime"].audit_hook = audit
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        collect(tx, case)
    assert seen == [(group, count) for group in range(16) for count in range(1, 9)]


def test_rollout_to_real_cpu_adam_checkpoint_schema_integration(tmp_path, prototype):
    case = prepare(tmp_path, prototype)
    runtime, optimizer, scheduler = case["runtime"], case["optimizer"], case["scheduler"]
    with RunTransactions(case["run"], case["identity"]) as tx:
        start(tx, case)
        ready = collect(tx, case)
        before, rng = state_hash(trainable_state(runtime.model)), state_hash(capture_rng())
        # Real tiny Adam serialization, with explicitly synthetic probability
        # evidence. This verifies schema integration, never the production loss.
        optimizer.zero_grad(set_to_none=True)
        runtime.model(torch.ones(2)).square().sum().backward()
        torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update = dict(
            status="UPDATED_UNCOMMITTED",
            optimizer_updates=1,
            sequence_count=128,
            group_sizes=[8] * 16,
            actual_credit_groups=ready["credits"],
            policy_before_sha256=before,
            policy_after_sha256=state_hash(trainable_state(runtime.model)),
            reference_sha256=case["identity"]["reference_sha256"],
            rng_restored=True,
            rng_before_sha256=rng,
            rng_after_sha256=rng,
            fixture_only=True,
            probability_vectors=[
                dict(
                    row_id=r["row_id"],
                    qid=r["qid"],
                    tokens=r["tokens"],
                    sampler_logprobs=r["sampler_logprobs"],
                    train_logprobs=r["sampler_logprobs"],
                    reference_logprobs=r["sampler_logprobs"],
                    token_path_sha256=r["token_path_sha256"],
                )
                for r in ready["records"]
            ],
        )
        path = "rollouts/step-001/FIXTURE_PREUPDATE.json"
        evidence_hash = write(
            case["run"] / path,
            dict(
                status="PREUPDATE_PASS",
                run_id=runtime.identity["run_id"],
                committed_source_step=0,
                policy_before_sha256=before,
                fixture_only=True,
            ),
        )
        references = {**ready["references"], "evidence": dict(path=path, sha256=evidence_hash)}
        tx.commit(
            runtime,
            optimizer,
            scheduler,
            step=1,
            sampler_state=ready["sampler_state"],
            update=update,
            references=references,
        )
        restored = tx.restore_latest(runtime, optimizer, scheduler)
        assert restored["step"] == 1 and restored["sampler_state"] == ready["sampler_state"]
        next_ready = collect(tx, case)
        assert next_ready["step"] == 2 and runtime.calls == 256
        assert set(next_ready["sampler_state"]["consumed_qids"]["1"]).isdisjoint(
            next_ready["sampler_state"]["consumed_qids"]["2"]
        )
