"""Real tokenizer/image processor evidence only; no model weights or forward calls."""
from pathlib import Path
import json
import sys
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from adapter.backend import prepare_public_input, file_sha, read_json, write_once


def main():
    bound = read_json(PACKAGE/'execution/RUN_BINDING.json')
    if file_sha(PACKAGE/'frozen/FREEZE.json') != bound['manifest_sha256']:
        raise ValueError('Frozen package changed')
    for ref in bound['implementation_files']:
        if file_sha(ref['path']) != ref['sha256']:
            raise ValueError('Bound implementation changed')
    sys.path.insert(0, bound['native_code_root']+'/src')
    from sr_f5.runtime import Runtime, NUMERICS_V2
    from mm_dev.data import reconstruct_rgb
    def rows(name):
        return [json.loads(l) for l in (PACKAGE/'frozen'/name).read_text().splitlines()]
    inputs = {r['input_id']:r for r in rows('inputs.jsonl')}
    cases = rows('private_cases.jsonl')
    chosen = []
    for count, long_prefix in [(8, False), (10, False), (10, True)]:
        candidates = [c for c in cases if len(c['task']['world']['categories']) == count
                      and (len(inputs[c['input_id']]['assistant_prefill']) > 1) == long_prefix
                      and not inputs[c['input_id']]['image_file'].startswith('legacy/')]
        chosen.append(sorted(candidates,key=lambda c:c['case_id'])[0])
    runtime = Runtime.processor_only(bound['model']['model_path'], protocol_revision='v2')
    runtime._srf5_numerics = dict(NUMERICS_V2)
    out = PACKAGE/'execution/PROCESSOR_CPU'
    out.mkdir(exist_ok=False)
    evidence = []
    for index, case in enumerate(chosen):
        inp = inputs[case['input_id']]
        image = PACKAGE/'frozen'/inp['image_file']
        request = dict(prompt=inp['prompt'], assistant_prefill=inp['assistant_prefill'],
                       image_file=str(image.relative_to(PACKAGE)), image_sha256=file_sha(image))
        prepared = prepare_public_input(runtime, request, PACKAGE)
        routing = prepared['routing']
        rgb = reconstruct_rgb(prepared['inputs']['pixel_values'], routing['image_grid_thw'], runtime.processor.image_processor)
        output = out/f'processed-{index}.png'
        rgb.save(output)
        evidence.append(dict(case_id=case['case_id'], input_id=case['input_id'],
            categories=len(case['task']['world']['categories']),
            source_file=str(image), source_sha256=file_sha(image),
            processed_file=str(output), processed_sha256=file_sha(output),
            assistant_prefill=inp['assistant_prefill'], routing=routing,
            actual_prompt_tokens=prepared['inputs']['input_ids'][0].tolist()))
    write_once(out/'RECEIPT.json', dict(status='PROCESSOR_CPU_COMPLETE_VISUAL_PENDING',
        evidence=evidence, model_weights_loaded=False, model_calls=0, optimizer_updates=0,
        script_sha256=file_sha(__file__), frozen_manifest_sha256=bound['manifest_sha256']))
    print('PROCESSOR_CPU_COMPLETE_VISUAL_PENDING: 8/10 categories and long prefill; zero model calls')


if __name__ == '__main__':
    main()
