import json
from pathlib import Path
nmodels=16
parts={
 'scientific_training':15*96*16*8,
 'monitor_baseline_plus_endpoints':nmodels*512*8,
 'train_fit_baseline_and_steps32_64_96':(1+15*3)*32*4,
 'test_ID_two_protocols':nmodels*2048*4*2,
 'three_shift_panels_two_protocols':nmodels*3*128*4*2,
 'five_diagnostic_views':nmodels*64*5*8,
 'ChartQA_expected_2500':nmodels*2500,
}
assert sum(parts.values())==648000
out={'planned_generated_completions':parts,'total_if_ChartQA_has_2500':sum(parts.values()),
 'without_external':sum(parts.values())-parts['ChartQA_expected_2500'],
 'base_format_check':32*8,'post_bridge_format_check_if_used':32*8,'post_bridge_paired_format_recheck':32*8,
 'natural_engine_generations':8*16*8,'optional_nonzero_kernel_generations':8*16*8,
 'common_bridge_updates_if_used':16,'scientific_updates':15*96,
 'single_run_rollouts':96*16*8,'additional_teacher_and_probe_teacher_forcing':'Record separately, not generations',
 'gpu_hours_limit':None,'wallclock_stop_limit':None,
 'actual_runtime_measurements':False}
Path('validation/BUDGET.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
