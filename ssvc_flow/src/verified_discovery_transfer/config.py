"""Protocol configuration and small, deterministic metadata identities."""

import hashlib
import json
from pathlib import Path

DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[2] / "docs/verified_discovery_transfer/design/protocol.json"
)
TEMPLATE_VERSION = "inherit-O0-L11-B1-plus-new-cohort-v1"
PROTOCOLS = {
    "trend": ("O0", "L11"),
    "cross_series": ("O0", "L11", "B1"),
    "duplicate_encoding": ("O0",),
}
COMPANION = {"trend": "L11", "cross_series": "B1"}
PROTOCOL_ORDER = ("O0", "L11", "B1")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(path=None):
    config = json.loads(Path(path or DEFAULT_PROTOCOL).read_text())
    if config.get("schema") != "ssvc-verified-discovery-transfer-1":
        raise ValueError("Unknown registered protocol schema")
    # This executable implements this frozen experiment, not a hyperparameter search API.
    registered = json.loads(DEFAULT_PROTOCOL.read_text())
    for section in (
        "model",
        "generation",
        "data",
        "protocols",
        "discovery",
        "sft",
        "R0_reference",
        "evaluation",
        "statistics",
    ):
        if config[section] != registered[section]:
            raise ValueError(
                f"Unregistered change to {section}; requires a new protocol implementation"
            )
    if (
        type(config["hardware"]["max_gpus"]) is not int
        or not 1 <= config["hardware"]["max_gpus"] <= 5
    ):
        raise ValueError("At most five GPUs")
    return config
