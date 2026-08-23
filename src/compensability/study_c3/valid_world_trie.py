"""Tokenizer-aware finite-state decoder for the 17^4 valid Study C3 worlds."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class TokenizerLike(Protocol):
    eos_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(slots=True)
class _Node:
    children: dict[int, int]
    terminal: bool = False


class ValidWorldTrie:
    """Prefix trie over tokenizer encodings, with EOS as the only terminal transition."""

    def __init__(
        self,
        *,
        nodes: list[_Node],
        eos_token_id: int,
        minimum: int,
        maximum: int,
        world_count: int,
    ) -> None:
        self._nodes = nodes
        self.eos_token_id = eos_token_id
        self.minimum = minimum
        self.maximum = maximum
        self.world_count = world_count

    @classmethod
    def build(
        cls, tokenizer: TokenizerLike, *, minimum: int = 2, maximum: int = 18
    ) -> ValidWorldTrie:
        eos = getattr(tokenizer, "eos_token_id", None)
        encode = getattr(tokenizer, "encode", None)
        if type(eos) is not int or not callable(encode):
            raise ValueError("Study C3 tokenizer requires integer EOS and encode support")
        if type(minimum) is not int or type(maximum) is not int or minimum > maximum:
            raise ValueError("Study C3 world domain is invalid")
        nodes = [_Node({})]
        world_count = 0
        for values in itertools.product(range(minimum, maximum + 1), repeat=4):
            action = ",".join(str(value) for value in values)
            token_ids = encode(action, add_special_tokens=False)
            if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
                raise ValueError(f"tokenizer returned invalid IDs for legal action {action}")
            node_index = 0
            for token in token_ids:
                node = nodes[node_index]
                node_index = node.children.setdefault(token, len(nodes))
                if node_index == len(nodes):
                    nodes.append(_Node({}))
            if nodes[node_index].terminal:
                raise ValueError("tokenizer collision maps distinct legal worlds to one token path")
            nodes[node_index].terminal = True
            world_count += 1
        if any(node.terminal and node.children for node in nodes):
            raise ValueError(
                "tokenizer creates a terminal-prefix collision in the legal world language"
            )
        return cls(
            nodes=nodes,
            eos_token_id=eos,
            minimum=minimum,
            maximum=maximum,
            world_count=world_count,
        )

    def iter_actions(self) -> Iterable[str]:
        for values in itertools.product(range(self.minimum, self.maximum + 1), repeat=4):
            yield ",".join(str(value) for value in values)

    def _node_for(self, prefix_token_ids: Sequence[int]) -> int:
        node_index = 0
        for token in prefix_token_ids:
            child = self._nodes[node_index].children.get(int(token))
            if child is None:
                raise ValueError("invalid constrained-decoding prefix")
            node_index = child
        return node_index

    def allowed_next(self, prefix_token_ids: Sequence[int]) -> frozenset[int]:
        node = self._nodes[self._node_for(prefix_token_ids)]
        if node.terminal:
            return frozenset({self.eos_token_id})
        if not node.children:
            raise ValueError("constrained-decoding prefix reaches a dead state")
        return frozenset(node.children)

    def accepts(self, token_ids: Sequence[int]) -> bool:
        values = list(token_ids)
        if values and values[-1] == self.eos_token_id:
            values = values[:-1]
        try:
            node = self._nodes[self._node_for(values)]
        except ValueError:
            return False
        return node.terminal

    def mask_logits(
        self, prefix_token_ids: Sequence[int], logits: Sequence[float]
    ) -> tuple[list[float], dict[str, object]]:
        values = [float(value) for value in logits]
        if not values or any(math.isnan(value) for value in values):
            raise ValueError("Study C3 logits must be non-empty and non-NaN")
        allowed = self.allowed_next(prefix_token_ids)
        if any(token >= len(values) for token in allowed):
            raise ValueError("Study C3 allowed token lies outside the logits vocabulary")
        before = max(range(len(values)), key=values.__getitem__)
        masked = [
            value if index in allowed else float("-inf")
            for index, value in enumerate(values)
        ]
        after = max(range(len(masked)), key=masked.__getitem__)
        record = {
            "prefix_token_ids": [int(token) for token in prefix_token_ids],
            "allowed_token_ids": sorted(allowed),
            "pre_mask_argmax_token_id": before,
            "post_mask_argmax_token_id": after,
            "pre_mask_finite_count": sum(math.isfinite(value) for value in values),
            "post_mask_finite_count": sum(math.isfinite(value) for value in masked),
            "masked_token_count": len(values) - len(allowed),
        }
        return masked, record

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "status": "STUDY_C3_VALID_WORLD_TRIE_COMPLETE",
            "value_domain": [self.minimum, self.maximum],
            "world_count": self.world_count,
            "eos_token_id": self.eos_token_id,
            "node_count": len(self._nodes),
            "nodes": [
                {
                    "terminal": node.terminal,
                    "children": {str(key): value for key, value in sorted(node.children.items())},
                }
                for node in self._nodes
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ValidWorldTrie:
        domain = payload.get("value_domain")
        raw_nodes = payload.get("nodes")
        if (
            payload.get("schema_version") != 3
            or payload.get("status") != "STUDY_C3_VALID_WORLD_TRIE_COMPLETE"
            or not isinstance(domain, list)
            or len(domain) != 2
            or not isinstance(raw_nodes, list)
        ):
            raise ValueError("Study C3 trie payload is malformed")
        nodes: list[_Node] = []
        for raw in raw_nodes:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("children"), Mapping):
                raise ValueError("Study C3 trie node is malformed")
            nodes.append(
                _Node(
                    {int(key): int(value) for key, value in raw["children"].items()},
                    raw.get("terminal") is True,
                )
            )
        result = cls(
            nodes=nodes,
            eos_token_id=int(payload["eos_token_id"]),
            minimum=int(domain[0]),
            maximum=int(domain[1]),
            world_count=int(payload["world_count"]),
        )
        if result.world_count != (result.maximum - result.minimum + 1) ** 4:
            raise ValueError("Study C3 trie world count drifted")
        return result


__all__ = ["TokenizerLike", "ValidWorldTrie"]
