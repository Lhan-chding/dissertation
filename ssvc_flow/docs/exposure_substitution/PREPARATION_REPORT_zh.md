# SER-J2 准备记录

本轮按用户明确授权实施固定共同训练槽位下的结构供体替换；附件中的历史建议仅为背景，执行范围以入口和主方案为准。无新奖励、RL、发现矩阵或剂量扫描。

## 来源和身份

独立分支为 `codex/structural-exposure-replacement-20261007`，基于 VDT `e375caa9118645f5f93aa76705f634c971e1fb7d`。它与原真实执行提交 `ed1d24839dc6bd374d17a92bd29678b7185c7605` 在 VDT、protocol-state-probes、prospective-selection 运行源码上的 diff 为空。

原始运行服务器根为 `/projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006/run_v2`。已实际读取并重算两个父检查点完整文件哈希，均符合原绑定；未重建父模型或修改旧产物。见 [服务器来源核验](evidence/SERVER_SOURCE_INVENTORY.json)。

共同语料由四个 SELF_O0 focus 视图交集得到194题，排除供体单元12题后保留182题。四个原视图的目标、公开渲染和训练签名均可从真实 discovery 记录复现；验证不仅依赖 audit 真值。64条回放沿用原 verified_targets 来源。全部246条复用目标通过公开唯一修复验证。

## 冻结数据与限制

32个供体根各有A/B/C三种结构，共96条输入；同根观察、j2、操作和规范目标相同。新诊断48题、确认800题，全部944条新任务通过唯一公开修复验证。共274个新根，DONOR_TRAIN/E_DIAG/E_CONFIRM无根或真值多重集交集。确认c3j3/c4j4各128题，其余cross单元各32题。

历史排除总计4219个orbit。补充检查覆盖本地现有SSVC worktree的artifacts/data目录和服务器louis-ssvc任务清单：10个本地清单、28个服务器清单均属于已归档集合，新增orbit为0。具体范围和未覆盖的外部存储限制见[后续历史检查](evidence/LATER_HISTORY_AUDIT.json)。

供体数值互异、真值域0..99、全部SYMBOLIC_FRESH；A是有学习作用的参照。共同背景仍可包含其他前位叶题，不将其描述为纯净零曝光背景。32个供体根跨父与题序固定，结论条件于该集合。

三份明确schedule使用108701/108702/108703。每个学生256步，每步11 common＋1 donor＋4 replay，总4096条目标曝光。供体单独microbatch，五组为4/4/3/1/4，每条序列权重1/16。共同task_id和供体root_id严格同step/slot。重建任务、目标、schedule、作业和sentinel记录与附件一致。

## 状态边界

本地准备产物位于工作树 `artifacts/ser_j2_20261007`；本地CPU数据验收通过。服务器须从实际旧运行目录重新prepare并绑定外部runtime/protocol后，才允许模型调用。单次真实GPU恢复桥接必须完成且绑定同一冻结计划，随后运行全部18个预定学生及95个评估作业。E_CONFIRM只在全部登记终点终态且分析代码未变化时统一揭晓。

正式预算为4608更新、73728目标曝光、145280生成回答。技术桥接上限16更新、256回答，单列记账。CPU测试通过不代表GPU执行完成，也不构成科学结论。
