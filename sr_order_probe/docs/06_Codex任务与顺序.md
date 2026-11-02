# 06｜Codex实施工作单：一次预检、一次烟测、一次主探查

## 总要求

本轮不再循环“更多审计—再发方案”。只完成以下有限产物。没有新训练授权，没有效果正向门，没有自动重设计。

| 任务 | 实施内容 | 必须交付 |
|---|---|---|
| T0 身份绑定 | 新目录；读取原父、snapshot、原FP32生成器/processor版本；查看当前队列 | `SOURCE_BINDING.json`，实际路径/哈希/版本；绝不复用旧FAIL的PASS签名 |
| T1 数据与提示 | 使用本包固定输入；双oracle、命名/前缀/置换核验；历史已用世界排重；实际processor图像检查 | `INPUT_VALIDATION.json`，prompt差分、图像检查记录、完整SHA manifest |
| T2 只读后端 | 绑定长assistant prefill，真实suffix token与stop状态；禁止optimizer/backward | `RUNTIME_BINDING.json`，最小代码差分和CPU夹具回归 |
| T3 独立烟测 | 全部216条，主题与烟测同时预先冻结；不按表现改提示 | 原始216条与 `SMOKE_RECEIPT.json`，技术检查、UNKNOWN/长度描述 |
| T4 主探查 | 4,320条、三父全部执行，request级幂等恢复 | 全部原输出、attempt日志、真实成本、`MAIN_COMPLETE.json` |
| T5 评分与统计 | 所有分配输出；72条有限独立复核；配对根分析 | 评分逐行、分母表、8个主比较、次比较、案例与限制 |
| T6 交付并停止 | 只读打包，不启动训练/新推理分支 | 中文报告、JSON/CSV、最小可重跑分析；后续建议标为未授权 |

## T0/T2：既有代码如何复用

原包中：

- `06_SOURCE/sr_f6_code_b7e7b06/src/sr_f6/interface_v3_protocol.py`是提示来源；
- `06_SOURCE/.../src/sr_f6/measurement.py`是原生评分适配来源；
- `06_SOURCE/sr_f6_dev_eval_v2_code_aeda61b/scripts/sr_f6_dev_eval_v2/runtime_adapter.py`是只读加载/评估参考；
- 实际生成kernel来自运行环境中的已绑定native Runtime。

Codex应复用模型加载、processor、采样、FP32约束和原评分实现；**新建轻量只读实验driver**，不要把新数据写进原 `PILOT_V3` / `FORMAL` 的身份文件，不重建数千行批准系统，不调用训练恢复模块。

需要自己完成、此包没有声称已完成的函数：

```python
make_backend(config, parent, run_dir) -> BoundReadOnlyBackend
backend.generate_one(public_request) -> raw_output_record
backend.identity() -> actual_loaded_identity
backend.close()
```

`backend/interface.py`给出输入输出契约并故意拒绝空实现。若生产引擎不支持长assistant prefill，应修复该接口、用烟测验证；不能改成user注入、先生成再剪贴或普通新assistant消息，这会改变干预本身。

## 固定技术通过条件

真实identity匹配；实际输入与manifest一致；没有私有gold进入prompt；prefix事实正确；prefix确实进入同一assistant上下文；记录采样token及真实终止；状态机不会提前停止；0更新；CPU评分夹具一致；输出按ID完整且不重复。

这些条件与模型答对率、S/B通过率、L服从率、效应方向无关。模型实际不服从指令或生成非法JSON，不可被当成后端错误而重采。

若烟测无法证明长前缀真实生效，阻止主探查并输出具体接口失败，**不自动删掉三个前缀条件**。如果仅表现差，保留并按原计划继续；若发现实质模型不可测性，则报告测量限制，不擅改成新评分。

## 冻结与修改纪律

在烟测首条生成前冻结：本方案、config、全部主/烟测world/query、图像、文本、前缀、seed及分析代码。技术绑定可以根据烟测修复可定位的实现错误，但要记录修复版本；科学内容不能看到烟测输出后修改。

如需改题目/提示语义/采样参数/样本量/主要指标：停止旧版本，提交另立版本建议，不覆盖原件，不在本轮自动搜索多个方案。历史FAIL保持原样。

## 本地与服务器命令

本地CPU命令已可运行：

```bash
python -m unittest discover -s tests -v
python scripts/validate_inputs.py --inputs data --out qa/input_check.json
# 给另一个空目录重建；不可覆盖已经冻结的data：
python scripts/build_inputs.py --out /path/to/new_empty_dir --render --font /path/to/installed_font.ttf
python scripts/check_historical_overlap.py --inputs data --archive-dir /path/to/uploads --out qa/history_check.json
```

生产driver由Codex完成并以该新实验为入口，不提供虚假的已运行命令。建议CLI合同：

```text
python -m order_probe_server prepare --config config/plan.json --inputs data --run-root <new_root>
python -m order_probe_server run --split SMOKE --parent b0|b1|b2
python -m order_probe_server seal-smoke
python -m order_probe_server run --split MAIN --parent b0|b1|b2
python -m order_probe_server collect --split MAIN
```

这些是**需实现的接口名称**，不是已存在于服务器的程序。运行前必须输出实际实现与版本，不能调用不存在的命令后宣称完成。

已有原文落盘后，可使用本包：

```bash
python scripts/score_answers.py --inputs data --answers <main_raw_answers.jsonl> --split MAIN --out <new_scores.jsonl>
python scripts/analyze.py --scores <new_scores.jsonl> --out <new_analysis_directory>
```

`check_ready.py`只是有限技术清单核验；布尔回执不是事实证明，必须指向实际runtime/输入/烟测记录。模板保持NOT_RUN，不可手填以绕过检查。
