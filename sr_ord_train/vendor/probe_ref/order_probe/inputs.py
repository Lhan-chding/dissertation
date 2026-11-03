"""Exact condition manipulations, truthful prefill, and public-only requests."""
from __future__ import annotations
import copy, json, random
from pathlib import Path
from .common import digest, seed_for, text_sha

NATURAL = ('C0','C_NOLIST','C_ENUM','C_REF','C_PRINT','C_ALT')
PREFIX = ('P_STABLE','P_REVTIE','P_PLACEBO')
TIE_SENTENCE = 'Break ranking ties by printed left-to-right order.'
REF_SENTENCE = ('Break ranking ties by the category positions on the chart horizontal axis, '
                'from left to right. The order in which you write evidence does not change '
                'these chart positions.')
ORDER_INSTRUCTIONS = {
 'C_PRINT': 'For the Alpha evidence entries, list categories in printed left-to-right order. Keep the question\'s tie-breaking rule unchanged.',
 'C_ALT': 'For the Alpha evidence entries, list categories in alphabetical order by category label. Keep the question\'s tie-breaking rule unchanged.'}


def question_text(task: dict) -> str:
    q = task['query']
    if q['family'] != 'TOPK' or q['selection_series'] != 'Alpha' or q['value'] != 'Beta':
        raise ValueError('This probe does not silently extend its query grammar')
    return (f"Select the {q['k']} categories with the {q['direction']} Alpha values. "
            f"Compute the {q['aggregate']} of Beta for the final selected categories. "
            'All chart quantities are integer counts in [10,90]. ' + TIE_SENTENCE + ' ')


def prompt(task: dict, condition: str, format_contract: str) -> str:
    if condition not in NATURAL + PREFIX: raise ValueError('Unregistered condition')
    natural_condition = 'C_NOLIST' if condition in PREFIX else condition
    labels = task['world']['categories']
    text = question_text(task)
    if natural_condition == 'C_REF': text = text.replace(TIE_SENTENCE, REF_SENTENCE)
    text += '\n' + format_contract
    lines = ['EVIDENCE CHECKLIST FOR THIS QUESTION:']
    if natural_condition == 'C_NOLIST':
        lines.append('- Ranking: report the original Alpha value for EVERY category shown on the chart.')
    else:
        if natural_condition == 'C_ENUM':
            order = [labels[i] for i in task['enum_permutation']]
        else:
            order = labels
        lines.append('- Ranking: report the original Alpha value for EVERY category '
                     f'in this public list: {json.dumps(order, ensure_ascii=False)}.')
    lines.extend([
        '- Operand: report the original Beta value of each finally selected category in evidence.',
        '- A cell already reported for ranking or filtering is reported only once. Values written only in bindings or work do not count as evidence.'
    ])
    text += '\n' + '\n'.join(lines)
    if natural_condition in ORDER_INSTRUCTIONS:
        text += '\n' + ORDER_INSTRUCTIONS[natural_condition]
    return text


def prefix_orders(task: dict) -> dict[str, list[int]]:
    w, q = task['world'], task['query']
    sign = -1 if q['direction'] == 'largest' else 1
    base = sorted(range(len(w['categories'])), key=lambda i: (sign * w['series']['Alpha'][i], i))
    def reverse_block(block: list[int]) -> list[int]:
        out = base.copy()
        slots = [j for j, i in enumerate(out) if i in block]
        vals = [out[j] for j in slots][::-1]
        for j, i in zip(slots, vals): out[j] = i
        return out
    return {'P_STABLE': base, 'P_REVTIE': reverse_block(task['boundary_indices']),
            'P_PLACEBO': reverse_block(task['placebo_indices'])}


def assistant_prefill(task: dict, condition: str) -> str:
    if condition in NATURAL: return '{'
    if condition not in PREFIX: raise ValueError('Unregistered prefix')
    order = prefix_orders(task)[condition]
    w = task['world']
    facts = [{'series':'Alpha','category':w['categories'][i], 'value':w['series']['Alpha'][i]} for i in order]
    # Supply all original Beta cells in alphabetical name order, held identical across
    # the three prefixes. Do not give a second printed-order list in Beta entries.
    # This is an explicit, common fact scaffold, not a natural-generation claim.
    facts += [{'series':'Beta','category':w['categories'][i], 'value':w['series']['Beta'][i]} for i in sorted(range(len(w['categories'])), key=lambda j: w['categories'][j])]
    return '{"evidence":' + json.dumps(facts, ensure_ascii=False, separators=(',', ':')) + ','


def prefix_choice(task: dict, condition: str) -> list[str]:
    """Diagnostic: task rank + supplied tie sequence, not a model-input field."""
    order = prefix_orders(task)[condition]
    return sorted(task['world']['categories'][i] for i in order[:task['query']['k']])


def request_rows(roots: list[dict], cfg: dict, format_contract: str) -> tuple[list[dict],list[dict]]:
    public, private = [], []
    for root in roots:
        for task in root['views']:
            private.append(copy.deepcopy(task))
            image_file = 'images/' + task['world_query_key'] + '.png'
            for parent in cfg['model']['parents']:
                for condition in NATURAL + PREFIX:
                    draws = 1 if task['split'] == 'SMOKE' else (2 if condition in NATURAL else 1)
                    kind = 'NATURAL' if condition in NATURAL else 'PREFIX'
                    for draw in range(draws):
                        # Common random numbers within each comparable branch. They do not
                        # imply a common hidden trajectory or independent root replicates.
                        seed = seed_for(cfg['data']['master_seed'], 'generation', task['split'], parent, task['qid'], kind, draw)
                        p = prompt(task, condition, format_contract)
                        prefill = assistant_prefill(task, condition)
                        rid = f"{task['qid']}/{parent}/{condition}/d{draw}"
                        row = {'request_id':rid,'qid':task['qid'],'root_id':task['root_id'],
                               'split':task['split'],'parent':parent,'condition':condition,
                               'branch':kind,'draw':draw,'seed':seed,
                               'image_file':image_file,'prompt':p,'assistant_prefill':prefill,
                               'prompt_sha256':text_sha(p),'prefill_sha256':text_sha(prefill),
                               'sampling':copy.deepcopy(cfg['generation'])}
                        public.append(row)
    # Frozen interleaving. Never group by success or compliance.
    random.Random(seed_for(cfg['data']['master_seed'], roots[0]['views'][0]['split'], 'request_order')).shuffle(public)
    if len({r['request_id'] for r in public}) != len(public): raise AssertionError('duplicate request')
    return public, private


def validate_prefix_triplet(task: dict) -> None:
    orders = prefix_orders(task)
    texts = {c: assistant_prefill(task,c) for c in PREFIX}
    objs = {c: json.loads(t[:-1] + '}')['evidence'] for c,t in texts.items()}
    facts = {c: sorted((e['series'],e['category'],e['value']) for e in obj) for c,obj in objs.items()}
    if len({json.dumps(v) for v in facts.values()}) != 1: raise ValueError('Prefix facts differ')
    if len({len(t) for t in texts.values()}) != 1: raise ValueError('Prefix character lengths differ')
    gold = sorted(task['gold']['selected'])
    if prefix_choice(task,'P_STABLE') != gold or prefix_choice(task,'P_PLACEBO') != gold:
        raise ValueError('stable/placebo prediction differs from gold')
    if prefix_choice(task,'P_REVTIE') == gold: raise ValueError('Boundary reversal did not change choice')
    # No special indication of targets or model answer in the supplied text.
    for t in texts.values():
        for field in ('"selected"','"bindings"','"answer"','"aggregate"','"work"'):
            if field in t: raise ValueError('Disallowed prefix field')
