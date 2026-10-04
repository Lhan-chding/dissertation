#!/usr/bin/env python3
"""Reproduce only existing-data tables; requires safe extracted first-four archive."""
import argparse,subprocess,sys,importlib.util
from pathlib import Path
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);a=p.parse_args()
if importlib.util.find_spec('numpy') is None:raise SystemExit('numpy required for saved kernel spectra; choose an existing Python environment with numpy')
for name,extra in [('opportunity_b.py',['--raw',str(a.raw)]),('support_cost.py',['--raw',str(a.raw),'--output',str(P)]),('selector_d.py',[]),('prestate_e.py',['--raw-root',str(a.raw)]),('finalize_scope.py',['--raw',str(a.raw)]),('verify_analysis.py',[])]:
 subprocess.run([sys.executable,str(P/name),*extra],cwd=P.parents[1],check=True)
