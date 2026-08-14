"""Complexity and alias bounds for shared experiment YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from compbias.io.yaml_config import load_yaml_mapping


def test_shared_yaml_loader_rejects_aliases(tmp_path: Path) -> None:
    config = tmp_path / "alias.yaml"
    config.write_text("source: &shared {value: 1}\ncopy: *shared\n", encoding="utf-8")

    with pytest.raises(ValueError, match="alias"):
        load_yaml_mapping(config)


def test_shared_yaml_loader_rejects_excessive_depth(tmp_path: Path) -> None:
    config = tmp_path / "deep.yaml"
    config.write_text("root: " + "[" * 65 + "0" + "]" * 65 + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"depth|complex"):
        load_yaml_mapping(config)


def test_shared_yaml_loader_rejects_excessive_node_count(tmp_path: Path) -> None:
    config = tmp_path / "nodes.yaml"
    config.write_text("items: [" + ",".join("0" for _ in range(100_001)) + "]\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"node|complex"):
        load_yaml_mapping(config)
