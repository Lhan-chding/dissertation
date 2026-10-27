import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from srf6_ref.gates import launch_check
p=argparse.ArgumentParser();p.add_argument('--receipts',type=Path,required=True);p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/experiment.json');a=p.parse_args()
r=launch_check(a.receipts,a.config);print(json.dumps(r,ensure_ascii=False,indent=2));sys.exit(0 if r['allowed'] else 2)
