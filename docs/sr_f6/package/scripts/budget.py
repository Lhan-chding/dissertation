import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from srf6_ref.planning import budget
root=Path(__file__).resolve().parents[1]
print(json.dumps(budget(json.loads((root/'configs/experiment.json').read_text())),ensure_ascii=False,indent=2))
