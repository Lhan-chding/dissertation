"""Make privilege-labelled diagnostic requests. Never mix this output into TRAIN.
neutral_hint is a template whose exact tokenizer-based padding is done by runtime.
"""
import json
from pathlib import Path
from semantic_contract import canonical,model_text

rows=[]
for t in map(json.loads,Path('manifests/TASKS_GOLD_AUDIT_ONLY.jsonl').read_text().splitlines()):
    if t['pool']!='MONITOR' or t['variant']!='v0':continue
    key_note='Required evidence object keys (no values): '+', '.join(e['series']+'/'+e['category'] for e in t['required_evidence'])+'.'
    values='Verified input measurements:\n'+'\n'.join(f"{e['series']}, {e['category']}: {e['value']} counts" for e in t['required_evidence'])
    table='Input data table (these are given facts, not model observations):\ncategory | Alpha | Beta\n'+'\n'.join(f"{c} | {t['world']['series']['Alpha'][i]} | {t['world']['series']['Beta'][i]}" for i,c in enumerate(t['world']['categories']))
    original=model_text(t['world'],t['query'],t['surface'])
    for view in ['keys_hint','gold_values','text_values','neutral_hint','blank_image']:
        text=original
        image=t['image_file']
        if view=='keys_hint':text+='\n\n'+key_note
        elif view=='gold_values':text+='\n\n'+values
        elif view=='text_values':text=table+'\n\n'+original;image=None
        elif view=='blank_image':image='images/diagnostic_blank.png'
        row={'diagnostic_id':t['qid']+'-'+view,'qid':t['qid'],'view':view,'image_file':image,'text':text,
             'pool':'MONITOR_DIAGNOSTIC_ONLY','allowed_in_training':False,
             'information_granted':{'keys_hint':'required_entity_keys','gold_values':'required_true_values','text_values':'full_true_table_no_image','neutral_hint':'no_extra_task_facts','blank_image':'no_visible_chart_values'}[view],
             'K':8}
        if view=='neutral_hint':
            row['runtime_neutral_padding']={'target_text':values,'candidate':'This is supplementary general text. Follow the question and use the required output structure. ', 'match_token_count_tolerance':1,'no_model_based_selection':True}
        rows.append(row)
assert len(rows)==320
Path('manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
print(json.dumps({'diagnostic_requests':len(rows),'models_planned':16,'generated_completions_planned':len(rows)*8*16}))
