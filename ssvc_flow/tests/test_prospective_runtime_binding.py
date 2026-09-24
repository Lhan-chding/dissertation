import copy
import json
from pathlib import Path

import pytest

from src.prospective_selection.protocol import load_protocol
from src.prospective_selection.runtime import check_inherited_contract, load_runtime


def inherited():
    path = Path(__file__).parents[1] / "configs/modeling_v4/protocol.json"
    return json.loads(path.read_text())


def test_inherited_measured_training_contract_and_literal_initialization():
    bridge = check_inherited_contract(load_protocol(), inherited())
    assert bridge["model"]["lora"]["r"] == 8
    assert "seed17" in bridge["initialization"]


@pytest.mark.parametrize(
    "section,key,value", [("generation", "top_k", 20), ("optimizer", "lr", 1e-4)]
)
def test_drift_requires_explicit_bridge(section, key, value):
    value_config = copy.deepcopy(inherited())
    value_config["qwen"][section][key] = value
    with pytest.raises(ValueError):
        check_inherited_contract(load_protocol(), value_config)


def test_no_model_load_without_execution_authorization():
    with pytest.raises(PermissionError):
        load_runtime(load_protocol(), "/does/not/exist")
