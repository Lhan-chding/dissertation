"""Explicit fake generation with actual differentiable CPU parameters and Adam."""

from types import SimpleNamespace

import torch

from src.core import canonical_hash


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(5))
        self.bias = torch.nn.Parameter(torch.zeros(5))
        self.frozen = torch.nn.Parameter(torch.ones(5), requires_grad=False)
        self.register_buffer("scale", torch.ones(5))
        self.child = torch.nn.Dropout(0.0)
        self.child.eval()


class FakeAdapter:
    execution_kind = "CPU_FAKE_TORCH"

    def __init__(self, seed=17):
        torch.manual_seed(seed)
        self.model = TinyPolicy()
        self.model_id, self.revision = "FAKE_TORCH_CPU", "0" * 40
        self.audit = {"processor_hash": "fake-processor", "tokenizer_hash": "fake-tokenizer"}
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=self.decode))
        self.eos_ids = {4}
        self.forward_calls = self.generation_calls = 0
        self.fail_after = None
        self.bad_scores = False
        self.rope_deltas = None

    @staticmethod
    def decode(tokens, **kwargs):
        return "".join(
            {0: "[1,2,3,4]", 1: "[2,1,3,4]", 2: "[9,2,3,4]", 3: "invalid", 4: ""}[t] for t in tokens
        )

    def _reset_positions(self):
        self.rope_deltas = None

    def prepare(self, prompt, root):
        image = prompt.get("image_path")
        return {
            "inputs": {"input_ids": torch.tensor([1, 2])},
            "image": image,
            "audit": {
                "enable_thinking": False,
                "final_prompt_hash": prompt["prompt_hash"],
                "tokenized_prompt_hash": canonical_hash(prompt["user"]),
                "input_tensor_hash": canonical_hash([1, 2]),
                "image_token_count": int(bool(image)),
                "pixel_values_hash": image,
            },
        }

    def logprobs(self, prepared, completion, require_grad=False):
        self.forward_calls += 1
        result = ((self.model.logits + self.model.bias) * self.model.scale).log_softmax(-1)[
            completion
        ]
        return result if require_grad else result.detach()

    def generate(self, prepared, *, seed, max_new_tokens, do_sample=True):
        if self.fail_after is not None and self.generation_calls >= self.fail_after:
            raise RuntimeError("injected generation interruption")
        self.generation_calls += 1
        generator = torch.Generator().manual_seed(seed)
        distribution = (self.model.logits[:4] + self.model.bias[:4]).softmax(-1)
        token = (
            int(torch.multinomial(distribution, 1, generator=generator))
            if do_sample
            else int(distribution.argmax())
        )
        tokens = [token, 4]
        scores = self.logprobs(prepared, tokens).tolist()
        if self.bad_scores:
            scores[0] = float("nan")
        return {
            "raw_completion": self.decode(tokens),
            "token_ids": tokens,
            "behavior_token_logprobs": scores,
            "completion_length": 2,
            "stop_reason": "eos",
            "vision_forward_calls": int(bool(prepared["image"])),
        }


def make_fake_adapter(seed=17):
    return FakeAdapter(seed)


def fake_prompts(n=2, split="control", interface=None):
    records = []
    for index in range(n):
        selected = interface or ("IMAGE_CUE_FRESH" if index % 2 else "SYM_CUE")
        scene_id = f"{split}-scene-{index // 2}"
        scene = {
            "base_scene_id": scene_id,
            "split": split,
            "truth_world": [1, 2, 3, 4],
            "observed_world": [9, 2, 3, 4],
            "changed_index": 0,
            "operation": "sum4",
            "chart_type": "line",
            "constraint_family": "duplicate_encoding",
            "cue": {"family": "duplicate_encoding", "known_index": 0, "known_value": 1},
            "image_path": f"{scene_id}.png",
            "image_hash": canonical_hash(scene_id),
        }
        prompt = {
            "user": f"{split} prompt {index}",
            "prompt_hash": canonical_hash([split, index]),
            "image_path": scene["image_path"] if selected == "IMAGE_CUE_FRESH" else None,
        }
        records.append(
            {
                "prompt_id": f"{split}-prompt-{index}",
                "base_scene_id": scene_id,
                "family": "duplicate_encoding",
                "interface": selected,
                "split": split,
                "prompt_hash": prompt["prompt_hash"],
                "scene_hash": canonical_hash(scene),
                "scene": scene,
                "prompt": prompt,
            }
        )
    return records


def warm_origin(adapter, step=64):
    from src.followup_updates import capture_state

    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    for _ in range(step):
        for parameter in adapter.model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter) * 0.01
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    origin = capture_state(
        adapter.model,
        optimizer,
        {"checkpoint_step": step, "arm": "X_BASE", "train_seed": 17},
        adapter=adapter,
    )
    return optimizer, origin
