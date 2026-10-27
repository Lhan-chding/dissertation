"""Arithmetic checks, not experimental results or power guarantees."""
import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from srf6_ref.credit import native_advantage,credit
from scipy.stats import t,norm
j=[1]*7+[0];a=native_advantage(j)
checks={'seven_success_advantage':a[0],'lone_failure_advantage':a[-1],
 'claude_f14_section3_issue':'source sets -0.378 for the lone J=0 in 7/8; correct value is -sqrt(7), and both negative terms cannot constitute a sign flip',
 'positive_success_local_coefficient_example':a[0]/400+(-.25*6/7)/20,
 'zero_of_32_one_sided95_upper':1-.05**(1/32),
 'wrong_consistency_residual_derivative_at_p01_lambda1':1-2*(1-.1)}
checks['power_sensitivity_not_empirical']=[]
for repeats in (2,3):
 for sigma_pp in (.5,1.,2.,4.):
  df=3*(repeats-1);se=sigma_pp/math.sqrt(3*repeats);critical=t.ppf(1-.05/(2*4),df)
  checks['power_sensitivity_not_empirical'].append({'independent_repeats_per_parent':repeats,'paired_run_sd_pp':sigma_pp,'df':df,
   'approx_MDE80_pp':float((critical+norm.ppf(.8))*se),'note':'normal power approximation plus small-sample critical; not a measured 72-step variance'})
print(json.dumps(checks,ensure_ascii=False,indent=2))
