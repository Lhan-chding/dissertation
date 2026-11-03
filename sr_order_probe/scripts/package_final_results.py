#!/usr/bin/env python3
"""Lossless bounded delivery of existing evidence; no model calls or deletions."""
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();r=a.results.resolve();src=a.source.resolve();capture=r/'SERVER_CAPTURE'
    delivery=r/'SR_ORDER_PROBE_COMPLETE_20261103'
    delivery.mkdir(exist_ok=False)
    def copy(source,target):
        if source.is_symlink():raise ValueError('Never follow symlink: '+str(source))
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir():
            for item in sorted(source.iterdir()):
                if item.name=='__pycache__' or item.suffix in ('.pyc','.ttf','.otf'):continue
                copy(item,target/item.name)
        else:shutil.copy2(source,target)
    for item in sorted(capture.iterdir()):
        if item.name=='RAW':continue
        copy(item,delivery/item.name)
    for split in ('SMOKE','MAIN'):
        original=capture/'RAW'/f'{split}.jsonl';target=delivery/'RAW'/f'{split}.jsonl.gz'
        target.parent.mkdir(exist_ok=True)
        with original.open('rb') as source,target.open('wb') as packed:
            with gzip.GzipFile(filename='',mode='wb',fileobj=packed,compresslevel=6,mtime=0) as output:
                shutil.copyfileobj(source,output)
        h=hashlib.sha256()
        with gzip.open(target,'rb') as check:
            for block in iter(lambda:check.read(1024*1024),b''):h.update(block)
        assert h.hexdigest()==sha(original)
    for item in sorted((r/'REPORT').iterdir()):copy(item,delivery/item.name)
    for name in ('FINAL_INTEGRITY.json','REPRODUCTION.json','COSTS.json','SCHEDULER_ACCOUNTING.psv','INTERPRETATION_REVIEW.md','NATIVE_SOURCE_20261103.tar.gz'):
        copy(r/name,delivery/name)
    copy(r/'AUDIT_REVIEW',delivery/'AUDIT/INDEPENDENT_REVIEW')
    copy(src/'execution',delivery/'EXECUTION/LOCAL_PREPARATION')
    for item in sorted(src.iterdir()):
        if item.name=='execution' or item.name.startswith('results_'):continue
        copy(item,delivery/'SOURCE_DIFF/sr_order_probe'/item.name)
    for split in ('SMOKE','MAIN'):
        copy(src/'data'/f'{split}_REQUESTS_PUBLIC.jsonl',delivery/'PUBLIC_REQUESTS'/f'{split}.jsonl')
        copy(src/'data'/f'{split}_TASKS_PRIVATE.jsonl',delivery/'PRIVATE_TASKS'/f'{split}.jsonl')
    provenance=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=src,text=True).strip(),
        source_status=subprocess.check_output(['git','status','--short','--',str(src)],cwd=src,text=True),
        original_user_zip_sha256='37c72507453d7203ebe11441d77dd5dfdcae07fcebb20a09508e156557a1052b',
        native_source_archive_sha256=sha(r/'NATIVE_SOURCE_20261103.tar.gz'),
        server_capture_sha256=sha(r/'DELIVERY_CAPTURE_20261103.tar.gz'))
    assert not provenance['source_status'],'Commit source before packaging'
    (delivery/'DELIVERY_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (delivery/'START_HERE_zh.md').write_text('''# SR-ORDER-PROBE 完整结果包

当前状态：216条烟测、4320条主推理、72条独立代理复核和中文报告均完成。本轮0训练更新；不再追加推理或实验。

1. 先读 RESULTS_REPORT_zh.md：三父结果、8项确认比较、前缀UNKNOWN与限制、成对反例、审核分歧。
2. CONDITION_SUMMARY.csv 是每父/合并×条件表；FULL_54_GROUPS.csv 是54组每父×条件×命名的合并描述表（A/S/V来自ANALYSIS_MAIN/DESCRIPTIVE.json，其余来自SUPPLEMENT_MAIN/SUPPLEMENTARY_DESCRIPTIVE.csv）；ANALYSIS_MAIN/CONTRASTS.json 是完整区间与检验。
3. RAW/MAIN.jsonl.gz 与 RAW/SMOKE.jsonl.gz 是完整无损原始输出（包含输入、前缀和suffix token、概率、真实身份），解压后分别4320/216行。gzip解压内容SHA与FINAL_INTEGRITY.json一致；不是只保留文本的精简投影。
4. AUDIT/INDEPENDENT_REVIEW 包含原文抽取、独立gold推导、初判、阶段哈希、比对与代理脚本；AUDIT_SUMMARY.json说明28处审核B编码分歧及裁定。72_selected_rows_and_findings.jsonl保留每条独立初判和冻结分数。
5. REPRODUCTION.json证明本地重新评分和统计与服务器字节一致。COSTS.json区分MAIN、SMOKE和初始化失败成本。
6. SOURCE_DIFF/sr_order_probe 是本轮源码/冻结数据/历史方案；NATIVE_SOURCE_20261103.tar.gz 是实际绑定的原生src。源码中的原包README/docs/qa历史NOT_RUN是制包时状态，不是当前执行状态。EXECUTION和WORKERS中的早期失败、NOT_RUN/排队快照也按历史保存；当前状态以本入口、最终报告、MAIN_COMPLETE与AUDIT_SUMMARY为准。不得把历史记录改写成成功。

报告和打包脚本是结果产生后的只读交付工具；它们不是事前登记分析。事前冻结的科学文件由SCIENCE_MANIFEST、INPUT_MANIFEST与SOURCE_DIFF/sr_order_probe/data/MANIFEST绑定。

原始运行根：/projects/_ssd/varunssd/louis-ssvc/sr_order_probe_20261102。完整模型权重、字体、可重建pycache、冗余逐请求RECORDS/ATTEMPTS_BY_ID副本以及其他项目均不进入包；相同内容的RAW/ATTEMPTS与WORKERS完整保留，服务器原件未删。本包不是含权重的训练恢复包。

## 复核步骤（CPU，不调用模型）

Python依赖见SOURCE_DIFF/sr_order_probe/requirements.txt。先解压RAW/MAIN.jsonl.gz至一个临时目录，再从SOURCE_DIFF/sr_order_probe运行：

```sh
python scripts/score_answers.py --inputs data --answers /absolute/temp/MAIN.jsonl --split MAIN --out /absolute/temp/SCORES_MAIN.jsonl
python scripts/analyze.py --scores /absolute/temp/SCORES_MAIN.jsonl --out /absolute/temp/ANALYSIS
```

所有输出目录应为新目录，不覆盖本包证据。只读复核无训练或新采样授权。不要执行服务器run命令重做已有回答。

MANIFEST_SHA256.json列出除自身以外每个文件的大小和SHA；zip已逐项读取校验。压缩包外另有DELIVERY_VERIFICATION.json和SHA256。
''')
    (delivery/'DATA_DICTIONARY_zh.md').write_text('''# 字段与分母

- parent：b0/b1/b2，分别为J6上游seed71001/71002/71003固定父。root_id为48个主世界聚类单位；qid含两命名视图；request_id还含条件/父/draw。
- 自然条件C0/C_NOLIST/C_ENUM/C_REF/C_PRINT/C_ALT每父192条，前缀P_STABLE/P_REVTIE/P_PLACEBO每父96条。自然每题2draw，前缀1draw；SMOKE独立，不混入MAIN。
- A：最终answer精确正确；A_parseable：答案可按冻结数值语法解析。不得从work补救answer。
- S：PASS=1、已知FAIL=0、UNKNOWN=null。S_hit仅PASS为1；其余0仅代表未确认正确。S_known、条件正确率及识别区间需同时查看。
- B：对任务正确集合的数值绑定正确性，三态；无法解析或缺失bindings为UNKNOWN，不能误写FAIL。V_probe是本轮形状有效性，不是原生J。
- complete：以balanced/eos为1、length等为0的生成终止指标，不等于JSON合法。json_valid独立提供。
- rank_order_known要求完整10个Alpha类别各一次；rank_values_true只在可判时评价数值真实性。rank_order_print/alphabetical/numeric_world可同时成立。
- order_compliance仅C_PRINT和C_ALT有语义；其他条件null不是缺失实验。forced_evidence=1代表前缀事实由实验给定，不能当模型自发证据真实性。
- reversed_boundary_choice_hit/enumeration_choice_hit/alphabetical_choice_hit是预设候选集合跟随事件，按全分母报告；应结合S_known解释。
- prompt_input_ids/suffix_token_ids/sampled_token_probabilities：真实输入/后缀token与选中token概率，prefill_boundary标出同assistant强制token边界。概率不是人工估算。
- generation_seconds的跨worker总和不是墙钟时长；GPU小时使用scheduler实际AllocTRES×ElapsedRaw，未测利用率/费用不得编造。
- CONTRASTS的estimate与区间单位是比例；报告表乘100为百分点。p为中心化配对cluster-bootstrap近似；只有固定三父合并的8项primary有p_holm_8。
- PAIRED_CASES是事后说明性例子，不是额外假设检验；72审核是预定有限实施复核，不是全体零错误认证。
''')
    files=sorted(p for p in delivery.rglob('*') if p.is_file())
    manifest=[dict(path=str(p.relative_to(delivery)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    (delivery/'MANIFEST_SHA256.json').write_text(json.dumps(dict(files=manifest,excludes=['MANIFEST_SHA256.json']),indent=2)+'\n')
    archive=r/(delivery.name+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(p for p in delivery.rglob('*') if p.is_file()):
            z.write(path,str(path.relative_to(delivery)),compress_type=zipfile.ZIP_STORED if path.suffix=='.gz' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names));assert z.testzip() is None
        expected={x['path']:x for x in manifest}
        for name in names:
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            if name in expected:
                h=hashlib.sha256();total=0
                with z.open(name) as f:
                    for block in iter(lambda:f.read(1024*1024),b''):h.update(block);total+=len(block)
                assert h.hexdigest()==expected[name]['sha256'] and total==expected[name]['bytes']
    receipt=dict(status='VERIFIED_COMPLETE_EVIDENCE_PACKAGE',archive=archive.name,bytes=archive.stat().st_size,
        sha256=sha(archive),members=len(names),manifest_entries=len(manifest),crc_read_to_eof=True,
        every_manifest_hash_verified=True,raw_gzip_roundtrip_verified=True,
        main_rows=4320,smoke_rows=216,independent_review_rows=72,optimizer_updates=0)
    (r/'DELIVERY_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (r/(archive.name+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
