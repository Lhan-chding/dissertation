# Selected CPU 初始化的单次提交

此辅助入口只提交一次 CPU 初始化。它不修改科学源码，不建立 selected 根，不启动训练或重跑失败的初始化。只有冻结 verify_selection 返回 PROTOCOL_SELECTED，并且选中的 selected_v2 或 selected_c2 根尚不存在，才允许注册、提交和执行。

它读取消融根已经部署的 operations/ablation_driver/7551617b2c3fa6c958be87059119d78a604580f6/，先核对已独立审查的 helper/test 精确哈希，再调用该 helper 认证科学 commit 652e52d530520a276b6420ff0f4ee479dcde6db1 的完整文件清单后导入冻结 API。不能用别的 dispatcher 或科学代码代替。

## 部署与调用

将本目录的 launcher、测试和验证记录部署到 NEWBASE/deployment/selected_cpu_launcher/。将独立审查并封存的完整初始化 Bash payload 放在 NEWBASE/deployment/selected_init/initialize_selected.sh，设置为只读 0444。其实际内容及所引用准备工具须由各自的准备 manifest 绑定；本 launcher 不解释或改变四阶段脚本内容。

    SELECTED_BASE=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016
    ABLATION_ROOT="$SELECTED_BASE/ablation"
    SELECTED_PY=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
    SELECTED_LAUNCHER="$SELECTED_BASE/deployment/selected_cpu_launcher/submit_selected_init.py"
    SELECTED_PAYLOAD="$SELECTED_BASE/deployment/selected_init/initialize_selected.sh"
    SELECTED_PAYLOAD_SHA=REPLACE_WITH_REVIEWED_64_HEX_SHA256

    "$SELECTED_PY" "$SELECTED_LAUNCHER" --mode register \
      --ablation-root "$ABLATION_ROOT" --python "$SELECTED_PY" \
      --payload "$SELECTED_PAYLOAD" --payload-sha256 "$SELECTED_PAYLOAD_SHA"

    "$SELECTED_PY" "$SELECTED_LAUNCHER" --mode submit --ablation-root "$ABLATION_ROOT"

register 创建不可覆盖的 NEWBASE/deployment/selected_init/OPERATION_IDENTITY.json，绑定选型回执、消融运行身份、科学源码 manifest、原 dispatcher 注册身份、此 launcher 原件及 payload 原件。相同注册可幂等读取；修改注册字段或文件内容会拒绝。

submit 与原消融提交入口共享 ablation/submissions/SUBMIT.lock。资源固定为 account rose、QoS soujanya-poria-startfund-2026-03、partition cluster02、node cpu-2、4 CPU、12 GiB、72 小时、0 GPU、no-requeue；作业名为 SRF3CE-SELECTED-INIT-LH。CPU 占一个本用户 teacher 提交名额。该入口核对共享 teacher 的实际、排队和未知 GPU 预留，以及本用户 MaxSubmit5。

提交前持久化 INTENT 和未知 STATE；sbatch 必须 held，核对真实 allocation 和 Slurm 保存的 batch script，再重新查询容量，持久化 RELEASE_INTENT 后释放。容量已满且尚未提交时返回 WAITING_FOR_CAPACITY，不建立 INTENT。只要 INTENT 存在，任何失败或未知结果都不允许再次 sbatch。

Slurm 实际执行 launcher 的 execute 模式。该模式再次核验完整选型、绑定文件、release 回执、真实作业、当前主机 cpu-2、零 GPU 和空 CUDA_VISIBLE_DEVICES，先写 STARTED，再只运行一次 /bin/bash <payload>。由 payload 的 static 阶段排他创建 selected 根。失败后保留原件与 partial，不重试。PAYLOAD_PROCESS_COMPLETED 只表示 shell 进程退出码 0，不表示 ENGINE 或科学实验验收。

## 只读调度查询与终态记录

    "$SELECTED_PY" "$SELECTED_LAUNCHER" --mode terminal --ablation-root "$ABLATION_ROOT"

此模式不调用 sbatch、release、cancel，也不执行 payload。作业仍在队列或 accounting 不明时只报告未完成；退出队列且真实 accounting 终态身份匹配后，写不可覆盖的 TERMINAL.json。它不以 CPU 成功终态代替准备阶段各自的审计回执。

本入口无自动续租、无第二序列、无自动失败重试。后续人工处理须先核对本次账本、真实 Slurm 终态和 partial 根；不得删除 INTENT 或 partial 后原样重投。

## 本地验证

从仓库根执行：

    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
      /Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m pytest \
      -q -p no:cacheprovider \
      artifacts/sr_f3_ce_protocol_fix_20261016/deployment/selected_cpu_launcher/test_submit_selected_init.py

测试只使用 fake Slurm、临时文件和冻结资源校验函数，无服务器、模型、GPU 或真实提交调用。最终记录见 LOCAL_VALIDATION.json。
