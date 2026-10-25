"""Model-visible text only. No gold sets, answers, or sidecar facts are consumed."""
CONTRACT = ('All chart quantities are integer counts in [10,90]. Category order means the printed left-to-right order, not alphabetical order. '
 'Break ranking ties using that printed order. Use signed Alpha minus Beta when a difference is requested. '
 'Return one JSON object with top-level fields "evidence", optionally "work", "selected", and "answer". '
 'Evidence is a list of objects with exactly "series", "category", and "value"; include enough true chart facts to verify selection and the answer. '
 'You may use a short work string to work through the problem; it is not required and has no prescribed internal format. '
 'The selected field must be an array of the category labels in your FINAL adopted set after every condition and filter; do not put a candidate pool there. '
 'The order within selected does not matter. Answer with an exact integer, exact finite decimal, or a quoted fraction such as "17/3". '
 'Do not use external tools or add text outside the JSON object.')

def question(world,query):
    value={'Alpha':'the Alpha values','Beta':'the Beta values','diff':'the signed differences (Alpha minus Beta)'}[query['value']]
    aggregate='sum' if query['aggregate']=='sum' else 'arithmetic mean'
    if query['family']=='TOPK':
        q=f"Select the {query['k']} categories with the {query['direction']} {query['selection_series']} values. Then compute the {aggregate} of {value} for those categories."
    else:
        cats=world['categories'];start,end=cats[query['start']],cats[query['end']]
        q=f"Consider the interval from {start} to {end} along the PRINTED LEFT-TO-RIGHT category positions. "
        q+=f"{'Include' if query['include_start'] else 'Exclude'} {start} and {'include' if query['include_end'] else 'exclude'} {end}. "
        if query['positive_only']:q+='Within that interval, keep only categories where Alpha is strictly greater than Beta. '
        q+=f'Compute the {aggregate} of {value} over the final selected categories.'
    return q

def prompt(world,query):return question(world,query)+'\n'+CONTRACT

def task_prompt(task):
    if task.get('protocol','structured_v3')=='natural_work_v2':
        return question(task['world'],task['query'])+'\n'+(
            'All chart quantities are integer counts in [10,90]. Use printed left-to-right order for ties. '
            'You may write concise working in natural language. End with exactly one <final>VALUE</final> tag '
            'containing the exact scalar answer, or null for an undefined empty aggregate. Do not use external tools.')
    return prompt(task['world'],task['query'])
