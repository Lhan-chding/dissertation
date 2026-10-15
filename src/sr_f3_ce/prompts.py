"""Frozen synthetic protocols; production prompts never consume gold sidecars."""

from __future__ import annotations

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
SEMANTICS = CONTRACT.split("Return one JSON object")[0].strip()
PROTOCOLS = ("structured", "answer_only", "natural_work")


def question_from_prompt(prompt: str) -> str:
    """Split a frozen input string, without consulting the audit task's question."""
    suffix = "\n" + CONTRACT
    if not isinstance(prompt, str) or not prompt.endswith(suffix):
        raise ValueError("Input does not have the frozen synthetic contract")
    question = prompt[: -len(suffix)]
    if not question or "\n" in question:
        raise ValueError("Frozen question must be one nonempty line")
    return question


def prompt_from_frozen(prompt: str, protocol: str = "structured") -> str:
    question = question_from_prompt(prompt)
    if protocol == "structured":
        return prompt
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


def generation_contract(protocol: str) -> dict:
    if protocol in ("structured", "answer_only"):
        return {"prefill": "{", "stop": "balanced_json", "max_new_tokens": 1024}
    if protocol == "natural_work":
        return {"prefill": "", "stop": "closed_final_tag", "max_new_tokens": 1024}
    raise ValueError("Unregistered synthetic protocol")
