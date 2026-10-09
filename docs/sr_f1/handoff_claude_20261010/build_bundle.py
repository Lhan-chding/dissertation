#!/usr/bin/env python3
"""Create the requested factual SR-F1 code/data handoff without remote operations."""
import argparse,csv,hashlib,importlib.util,io,json,os,re,shutil,subprocess,sys,zipfile
from pathlib import Path,PurePosixPath

NAME='SR_F1_CLAUDE_FACTS_AND_CODE_20261010'
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo)
def load_module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def stage(repo,capture,out):
 assert not out.exists(), 'Stage directory already exists; use a new path'
 out.mkdir(parents=True)
 docs=repo/'docs/sr_f1/handoff_claude_20261010'
 for p in sorted(docs.glob('*.md')):copy(p,out/p.name)
 for p in sorted((docs/'tools').glob('*.py')):copy(p,out/'tools'/p.name)
 copy(docs/'build_bundle.py',out/'tools/build_bundle_source.py')
 shutil.copytree(capture,out/'evidence')
 inv=json.loads(Path('/tmp/sr_f1_repair_source_inventory.json').read_text())
 source_files={}
 for rel,h in inv['files'].items():
  p=capture/'server/code/SOURCE_DEPLOYMENT.json' if rel=='SOURCE_DEPLOYMENT.json' else repo/rel
  b=p.read_bytes();assert sha(b)==h,rel
  target=out/'code'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b);source_files[rel]=h
 extra=['tests/mm_dev/test_cached_teacher_forcing.py','tests/mm_core/test_training.py','LICENSE']
 for rel in extra:copy(repo/rel,out/'code'/rel)
 (out/'code/README.md').write_text('SR-F1 deployed source snapshot e61e9de86e6954c94c751b6383b53da68718185b. See the handoff root START_HERE_FOR_CLAUDE_zh.md.\n')
 for p in sorted((repo/'docs/sr_f1').glob('*.md')):copy(p,out/'history/project_docs'/p.name)
 shutil.copytree(repo/'docs/sr_f1/validation',out/'history/project_docs/validation')
 shutil.copytree(repo/'docs/sr_f1/package',out/'history/project_docs/package',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
 diff=git(repo,'diff','c3001259acfb2347180bf9f9d62227c3997a3e40','e61e9de86e6954c94c751b6383b53da68718185b','--','src','scripts','tests')
 (out/'history/format_guard_repair.patch').write_bytes(diff)
 for rel in ['src/sr_f1/runtime.py','src/sr_f1/freeze.py','src/sr_f1/orchestration.py','scripts/sr_f1/submit_matrix.py']:
  p=out/'history/before_guard_repair'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(git(repo,'show','c3001259acfb2347180bf9f9d62227c3997a3e40:'+rel))
 mod=load_module(out/'tools/recompute_format_coverage.py','coverage_recompute');summary,rows=mod.compute(out)
 dump(out/'analysis/format_summary.json',summary)
 with (out/'analysis/FORMAT_RECORD_INDEX.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps({k:v for k,v in row.items() if k!='raw_text'},ensure_ascii=False)+'\n')
 tasks={r['qid']:r for r in map(json.loads,(out/'evidence/derived/TASKS_GOLD_FORMAT_BRIDGE.jsonl').read_text().splitlines())}
 inputs={r['qid']:r for r in map(json.loads,(out/'evidence/derived/MODEL_INPUTS_FORMAT_BRIDGE.jsonl').read_text().splitlines())}
 selected=[]
 for label in ['before','confirm']:
  panel=[r for r in rows if r['panel']==label]
  for kind in sorted({r['failure_type'] for r in panel}):selected.append(next(r for r in panel if r['failure_type']==kind))
  for truncated in [True]:
   candidates=[r for r in panel if r['truncated'] and r not in selected]
   if candidates:selected.append(candidates[0])
 lines=['# 原始回答示例','', '按面板、错误类别及文件名字典序选择首条，并各收录一条截断回答。全部回答均在原始记录目录；示例选择不改变统计分母。','']
 for n,r in enumerate(selected,1):
  inp=inputs[r['qid']];raw=json.loads((out/r['file']).read_text());fence='`'*(max([len(x) for x in re.findall(r'`+',r['raw_text'])]+[3])+1)
  lines += [f"## {n}. {r['panel']} / {r['failure_type']}",'',f"qid：`{r['qid']}`；draw：{r['draw']}；finish_reason：`{raw['finish_reason']}`；tokens：{len(raw['tokens'])}。",'',f"[完整原始记录](../{r['file']}) · [实际输入图像](../evidence/server/{inp['image_file']})",'', '输入文本：','',fence+'text',inp['text'],fence,'','原始回答（未清理）：','',fence+'text',r['raw_text'],fence,'']
 (out/'analysis/REPRESENTATIVE_CASES_zh.md').write_text('\n'.join(lines))
 metrics=[json.loads(p.read_text()) for p in sorted((out/'evidence/server/engineering/bridge/update_attempts').glob('*.json'))]
 with (out/'analysis/BRIDGE_UPDATES.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['logical_step','sequence_count','completion_tokens','mean_sequence_loss','minimum_sequence_loss','maximum_sequence_loss','gradient_norm','parameter_hash'])
  for r in metrics:w.writerow([r['logical_step'],len(r['qids']),sum(r['completion_tokens']),sum(r['sequence_losses'])/len(r['sequence_losses']),min(r['sequence_losses']),max(r['sequence_losses']),r['gradient_norm'],r['parameter_hash']])
 cap=json.loads((capture/'CAPTURE_MANIFEST.json').read_text())
 assert cap['raw_counts']=={'before':256,'confirm':256,'after':113}
 provenance={'package_kind':'FACTUAL_PROBLEM_CODE_AND_PARTIAL_EVIDENCE_SNAPSHOT','source_commit':inv['source_commit'],'source_tree_sha256':inv['source_tree_sha256'],'documentation_commit_at_build':git(repo,'rev-parse','HEAD').decode().strip(),'capture_started_at_utc':cap['capture_started_at_utc'],'capture_ended_at_utc':cap['capture_ended_at_utc'],'raw_counts':cap['raw_counts'],'deployed_source_files':source_files,'supplemental_local_files':{n:sha((repo/n).read_bytes()) for n in extra},'source_files_byte_identical':True,'model_calls_for_packaging':0,'server_mutations_for_packaging':0,'code_changes_for_training':False,'omitted':['base model weight tensors','zero/bridged LoRA adapter tensor files','full optimizer/RNG/checkpoint .pt payloads; JSON commit markers included','tokenizer vocabulary/tokenizer.json and merges; tokenizer config/chat template included','complete Python/CUDA environment; selected installed dependency source included','newly generated after records beyond the captured113','all unrelated experiments and credentials','ENGINE/science/final model evaluation outputs, which had not been generated'],'server_run_root':cap['run_root']}
 dump(out/'PROVENANCE.json',provenance)
 (out/'OMISSIONS_AND_PROVENANCE_zh.md').write_text('''# 收录、省略与来源

本包收录的是问题数据与源码快照。服务器根：`/projects/_ssd/varunssd/louis-ssvc/sr_f1_20261009`。取证区间为2026-10-10 00:12:02—00:12:06新加坡时间；各文件在区间内读取一次，日志/调度状态不是全服务器原子事务快照。

实际部署源码137个文件逐字节对照部署清单；另收录两个底层CPU测试、LICENSE和阅读说明。修复前源码与差异来自Git。服务器及实际依赖源码的每文件来源、大小、hash见 [CAPTURE_MANIFEST.json](evidence/CAPTURE_MANIFEST.json)，部署与打包身份见 [PROVENANCE.json](PROVENANCE.json)。

省略：基础模型权重、LoRA权重张量、完整Adam/RNG/checkpoint二进制、tokenizer词表与merges、完整Python/CUDA环境、无关实验及凭据。保留了checkpoint提交标记、身份hash、adapter配置和逐步训练记录。本包不能独立重训或加载9B模型。

after只收录113条，余下143条在取证时未收录；完整COVERAGE、COMMON_START和最终协议阻塞回执当时未生成。ENGINE、科学训练和最终模型评价尚未执行，没有对应输出。原任务包自带全量合成任务清单与gold标签，属于原合同输入文件。

历史项目文档和原包文件按原字节保留，其中较早的“正在运行/待执行”描述只对应各自时间点。包内新说明不包含后续修复建议。目录中的模型回答是原始数据。
''')
 (out/'EVIDENCE_INDEX_zh.md').write_text('''# 数据与错误信息索引

| 数据 | 收录位置 | 数量/状态 |
|---|---|---|
| 桥接前FORMAT | `evidence/server/engineering/format/before/` | 256/256；93条覆盖；22截断 |
| 独立FORMAT_CONFIRM | `evidence/server/engineering/format/confirm/` | 256/256；100条覆盖；17截断 |
| 桥接后原题复测 | `evidence/server/engineering/format/after/` | 113/256；部分快照 |
| 桥接逐步更新 | `evidence/server/engineering/bridge/update_attempts/` | 16步；全部128个gold completion/损失/梯度范数/hash |
| 检查点提交标记 | `evidence/server/engineering/bridge/checkpoints/` | step00—16；不含.pt张量 |
| 相关输入与真值 | `evidence/derived/` | FORMAT32、CONFIRM32、BRIDGE128 |
| 相关实际图像 | `evidence/server/images/` | 上述题目的图像集合 |
| 模型/模板配置 | `evidence/native_model_config/` | config、tokenizer_config、preprocessor_config、chat_template |
| 固定身份与处理路由 | `evidence/server/` | 冻结、模型环境、4800题processor路由 |
| 失败/恢复/调度记录 | `evidence/server/orchestration/` 与 `technical_incidents/` | 原失败attempt、恢复、取证时点任务状态 |
| 实际依赖源码 | `evidence/installed_dependency_source/` | Transformers5.14.1、PEFT0.19.1的部分实现 |

[独立原文复算结果](analysis/format_summary.json) · [记录索引](analysis/FORMAT_RECORD_INDEX.jsonl) · [16步训练数值表](analysis/BRIDGE_UPDATES.csv) · [原始回答示例](analysis/REPRESENTATIVE_CASES_zh.md)

确认面板错误分解：155条整段JSON/顶层协议无效，1条证据字段不可评分。模型生成状态全部COMPLETE、technical_validation_errors为空；“生成完成”与“格式达标”是不同字段。

原作业失败字符串为`FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW`、`ValueError('WORKER_RESULT_NOT_COMPLETE')`，见保留的日志/失败回执。桥接后确认覆盖100/256低于95%；源代码在原题复测完成后对应的门禁原因字符串为`ONE_BRIDGE_CONFIRMATION_BELOW_95_PERCENT`，该最终回执在取证时尚未生成。
''')
 print(json.dumps({'stage':str(out),'raw_counts':cap['raw_counts'],'server_files':len(cap['files']),'source_files':len(source_files)},indent=2))

def publish(out,destination):
 for p in out.rglob('__pycache__'):shutil.rmtree(p)
 for p in out.rglob('.pytest_cache'):shutil.rmtree(p)
 missing=[]
 for p in list(out.glob('*.md'))+[out/'analysis/REPRESENTATIVE_CASES_zh.md']:
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in link or link.startswith('#'):continue
   target=link.split('#')[0]
   if not (p.parent/target).is_file():missing.append({'document':str(p.relative_to(out)),'target':target})
 assert not missing,missing
 # Pattern scan for common credential formats; no environment/config secret files are selected.
 findings=[]
 patterns=[rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'AKIA[0-9A-Z]{16}',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'sk-[A-Za-z0-9]{32,}']
 for p in out.rglob('*'):
  if p.is_file():
   data=p.read_bytes()
   if any(re.search(pat,data) for pat in patterns):findings.append(str(p.relative_to(out)))
 assert not findings,findings
 files={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['MANIFEST.json','SHA256SUMS.txt']}
 dump(out/'MANIFEST.json',{'package':NAME,'files':files})
 hashes={n:r['sha256'] for n,r in files.items()};hashes['MANIFEST.json']=sha((out/'MANIFEST.json').read_bytes())
 (out/'SHA256SUMS.txt').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(hashes.items())))
 verifier=load_module(out/'tools/verify_bundle.py','bundle_verify');verified=verifier.verify(out)
 destination.mkdir(parents=True,exist_ok=True);final=destination/(NAME+'.zip');temp=destination/(NAME+'.partial.zip')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,NAME+'/'+p.relative_to(out).as_posix())
 with zipfile.ZipFile(temp) as z:
  assert z.testzip() is None
  names=z.namelist();assert len(names)==len(set(names))
  expected=set(hashes)|{'SHA256SUMS.txt'}
  assert {n[len(NAME)+1:] for n in names}==expected
  for n in names:
   p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts and '\\' not in n
   b=z.read(n);rel=n[len(NAME)+1:]
   if rel in hashes:assert sha(b)==hashes[rel],rel
   else:assert b==(out/'SHA256SUMS.txt').read_bytes()
  count=len(names);raw=sum(v.file_size for v in z.infolist())
 os.replace(temp,final)
 result={'status':'PASS','archive':str(final),'sha256':sha(final.read_bytes()),'bytes':final.stat().st_size,'members':count,'uncompressed_bytes':raw,'checks':['CRC and read to EOF','safe unique relative paths','all payload SHA256','manifest and checksum consistency','deployed source byte identity','server capture byte identity','recipient document relative links','credential pattern scan'],'raw_counts':json.loads((out/'PROVENANCE.json').read_text())['raw_counts'],'standalone_retraining_bundle':False,**verified}
 dump(Path(str(final)+'.verification.json'),result)
 Path(str(final)+'.sha256').write_text(result['sha256']+'  '+final.name+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['stage','publish']);p.add_argument('--repo',type=Path,required=True);p.add_argument('--capture',type=Path);p.add_argument('--stage',type=Path,required=True);p.add_argument('--destination',type=Path);a=p.parse_args()
 if a.mode=='stage':stage(a.repo.resolve(),a.capture.resolve(),a.stage.resolve())
 else:publish(a.stage.resolve(),a.destination.resolve())
