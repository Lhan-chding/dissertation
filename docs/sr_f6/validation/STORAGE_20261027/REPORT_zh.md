# 2026-10-27 本项目旧 raw 缓存清理回执

实际状态：`POSTFLIGHT_PASS`。仅清理 `/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_20261016/prior/raw` 中已无损归档的 45,454 个原文件；没有扩大到其它目录。完成于 2026-10-27 19:48（Asia/Singapore）。

|项目|实测值|
|---|---:|
|移除的旧 raw 文件|45,454|
|原文件逻辑字节|5,154,950,860|
|保留的压缩归档字节|549,096,352|
|新增归档与全部服务器审计文件（含最终核算回执）|602,763,209|
|源目录新增恢复索引|874 字节|
|扣除上述新增内容后的净逻辑占用减少|**4,552,186,777 字节（约 4.24 GiB）**|

文件 `st_blocks` 口径的净减少估计为 4,566,848,000 字节，尚未扣最后一份很小的核算回执；这不是对共享文件系统 `df` 变化的归因，快照、元数据和存储实现可能影响物理空闲空间。此次释放的是磁盘占用，没有清系统 page cache 或结束任何进程来制造“内存释放”。

## 保留位置与恢复

完整归档：
`/projects/_ssd/varunssd/louis-ssvc/storage_audits/cleanup_20261027_srf3_prior_raw/RAW_COMPLETE.tar.gz`

SHA-256：`a56b889a1d62e07563067ee09663675322dd3b71b8922b4de61100a1bc8dc37b`。

归档包含全部 45,454 个原始文件的安全、唯一、项目相对路径与完整字节。gzip 已读至 EOF 并校验 CRC，每个成员的字节数与 SHA-256 均匹配原文件。所有原目录保留，源目录现在只含 `ARCHIVED_LOCATION.json`。需要重新分析旧 raw 时，先从此归档恢复；不是把历史失败改成成功，也不是删除科学负结果。完整逐文件 `ALLOWLIST.json`、`INVENTORY.json` 与 `DELETE_JOURNAL.jsonl` 保存在同一服务器审计目录；本地保留较小的入口和验证回执。

可在原项目根恢复原文件，保留已有文件：

```bash
tar --keep-old-files -xzf /projects/_ssd/varunssd/louis-ssvc/storage_audits/cleanup_20261027_srf3_prior_raw/RAW_COMPLETE.tar.gz -C /projects/_ssd/varunssd/louis-ssvc
```

## 边界与验证

旧实验 `STOPPED_PROTOCOL_FAILURE` 状态保持原样。真实队列、当前登录节点打开句柄及 F5 配置依赖检查通过；已登记的新 F6 数据、模型与父 adapter 不依赖此目录。删除前重新核对完整源清单，逐文件使用不跟随 symlink 的目录描述符，仅移除与归档身份一致的精确 allowlist 成员。没有移除目录。

Qwen3.5-9B 缓存、三份 SRF2_J6 父 adapter、运行环境、F4 retention/ChartQA、F5 原文和失败证据、F5 step004/step012、旧 F3 `engine/resume` 全部保留。F5 运行器本来已自动删除其它事务步骤，不存在可直接再清的普通中间 step。

最初的两个 `cafeC-*` 活跃作业，以及后续 `cafeA/B/D-*` 作业，经可见 `Command`/日志路径确认属于 `/projects/_hdd/varunhdd/hgrel_hdd`，没有重命名或取消。最后快照中我们自己的新准备作业是 `229909 / LH-SRF6-Prepare / PENDING`，前缀符合要求。队列是瞬时观察，不代表后续状态；本清理没有提交 GPU 作业。

## 事故与避免重复校验

首次删除申请被自动审批拒绝，理由是审批未看到完整的归档完成及 STOP/依赖前置证据。该次没有执行删除；读出实际 `ARCHIVE_VERIFIED.json`、`PREFLIGHT.json`，补齐 `PREDELETE_PROOF.json` 后，相同受限动作获得自动审批接受。没有绕过审批。

两次连接落到 login-3，而初始归档在 login-1，最初的同节点守卫返回 exit 75，没有删除文件。实际对照发现只有项目挂载的 `st_dev` 从 43 变成 41；inode、size、mtime、ctime、mode、nlink、blocks 全相同。随后明确登记只允许这一挂载设备号映射：canonical 路径不变，当前文件 dev 必须等于当前项目 root dev，其余所有字段不变。相应回归测试已覆盖；内容身份仍来自已完成的逐成员 SHA 校验。

用户要求减少冗余 hash 时，审批补证用的第二次归档复验已经启动，因此如实保留该次记录。删除阶段和最终核算均复用已验证归档身份，检查源文件 stat，不再次读取 5.15GB 源内容；最终索引只计算小审计文件的 hash。完整经过见 `INCIDENTS.json` 和两份 exit75 日志。

## 本地实现验证

`scripts/sr_f6/storage_cleanup.py` 与服务器执行副本 SHA 一致：
`d82c838865b445b0dc568f5f9b9a1f312bec85bc78b8b243b64e13e84d95128b`。

针对路径越界、文件/父目录 symlink、源内容变化、坏归档/CRC、重复成员、验证复用和挂载设备号映射的 **21 项回归测试通过**；相关文件 Ruff 检查通过。上述是清理安全性测试，不能替代 SR-F6 模型或实验验收。
