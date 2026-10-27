"""Usage requires real exported records; this script never generates model evidence."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from srf6_ref.census import summarize
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
result=summarize(rows);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(a.output)
