"""Versioned synthetic protocols; prompts never consume gold sidecars."""

from __future__ import annotations

# V1 remains the default for preserved artifacts and historical audits.
CONTRACT = (
    "All displayed quantities are integer counts in [10,90]. Use the printed category order "
    "for ties. Empty sum or count is 0; other aggregates of an empty set are null. A "
    "one-element range is 0. Exact integers, finite decimals, and quoted fractions such as "
    '"98/3" are accepted. Return one JSON object with evidence and answer. Evidence is a '
    'list of {"series":...,"category":...,"value":...}. Include enough true facts to verify '
    "the selection and the answer; reporting all relevant facts is allowed. You may put a "
    "short work string after evidence and before answer, but it is optional. Do not use "
    "external tools."
)
CONTRACT_V1 = CONTRACT
CONTRACT_V2 = (
    "All displayed quantities are integer counts in [10,90]. Use the printed category order "
    "for ties. Empty sum or count is 0; other aggregates of an empty set are null. A "
    "one-element range is 0. Exact integers, finite decimals, and quoted fractions such as "
    '"98/3" are accepted. Do not use external tools. Include enough true chart facts to '
    "verify the selection and the answer; reporting all relevant facts is allowed. You may "
    'add a short "work" string after the evidence and before the answer; it is optional. '
    'Output exactly one JSON object whose top-level keys are "evidence", then optionally '
    '"work", then "answer". The value of "evidence" is a list; each list entry has exactly '
    'the keys "series", "category", and "value".'
)
SEMANTICS = CONTRACT_V1.split("Return one JSON object")[0].strip()
PROTOCOLS = ("structured", "answer_only", "natural_work")
REVISIONS = ("v1", "v2", "c2")
C2_PREFILL = '{\n  "evidence": ['


def contract_for_revision(revision: str = "v1") -> str:
    if revision not in REVISIONS:
        raise ValueError("Unregistered protocol revision")
    return CONTRACT_V2 if revision == "v2" else CONTRACT_V1


def question_from_prompt(prompt: str) -> str:
    """Split only a known complete frozen contract; never read an audit question."""
    if not isinstance(prompt, str):
        raise ValueError("Input does not have a known frozen synthetic contract")
    for contract in (CONTRACT_V1, CONTRACT_V2):
        suffix = "\n" + contract
        if prompt.endswith(suffix):
            question = prompt[: -len(suffix)]
            if not question or "\n" in question:
                raise ValueError("Frozen question must be one nonempty line")
            return question
    raise ValueError("Input does not have a known frozen synthetic contract")


def prompt_for_revision(prompt: str, revision: str) -> str:
    """Change only the structured contract for the registered ENGINE ablation."""
    return question_from_prompt(prompt) + "\n" + contract_for_revision(revision)


def prompt_from_frozen(prompt: str, protocol: str = "structured") -> str:
    question = question_from_prompt(prompt)
    if protocol == "structured":
        return prompt
    # These two protocols are the exact original V1 bytes for every revision.
    base = question + "\n" + SEMANTICS + "\n"
    if protocol == "answer_only":
        return base + (
            "Return one JSON object with only an answer field. Do not output evidence or a "
            "work field. Do not use external tools."
        )
    if protocol == "natural_work":
        return base + (
            "You may write concise working in natural language. End with exactly one "
            "<final>VALUE</final> tag containing the exact scalar answer, or null for an "
            "undefined empty aggregate. Do not use external tools."
        )
    raise ValueError("External protocols require their separately frozen author prompt")


def prompt_for(task: dict, protocol: str) -> str:
    """Compatibility interface; ONLY the independently frozen prompt is consumed."""
    return prompt_from_frozen(task["prompt"], protocol)


def prefill_for(protocol: str, revision: str = "v1") -> str:
    contract_for_revision(revision)
    if protocol == "structured":
        return C2_PREFILL if revision == "c2" else "{"
    if protocol == "answer_only":
        return "{"
    if protocol == "natural_work":
        return ""
    raise ValueError("Unregistered synthetic protocol")


def generation_contract(protocol: str, revision: str = "v1") -> dict:
    prefill = prefill_for(protocol, revision)
    if protocol in ("structured", "answer_only"):
        return {"prefill": prefill, "stop": "balanced_json", "max_new_tokens": 1024}
    return {"prefill": prefill, "stop": "closed_final_tag", "max_new_tokens": 1024}
