# 针对性验收清单

参考测试通过不等于以下仓库/真实模型测试已通过。按模块验收一次；模块改变后只重跑相关测试。

## A. 数据和协议

- [ ] 新T/V/E的family×j、cross center×j配额正确；0–99真值分布明确标注，与历史0–49区分。
- [ ] 单错解唯一，bool/float/额外解释不通过；纯越界整数可保留诊断但主类I。
- [ ] 历史暴露和新split以保守truth orbit隔离；同场景的图像guard与E明确相关。
- [ ] 实例生成不读取teacher结果；T、V、E、R文件权限由角色控制，非只靠文件名。
- [ ] L11只固定逆排列一次；difference_pairs使用规范坐标。
- [ ] B1不读取j/真值；解集不变，C_display和C_orig分别保存。
- [ ] 每题public verifier与隔离solver的X一致；不能在SELF构造时从solver填空。

## B. 发现与预算

- [ ] T每协议16条×2repeat；V/E每协议16条；每个parent独立teacher固定不更新。
- [ ] MIX仅取8O0+8companion；不是16+8。部署策略费用与整个研究原子库费用分别统计。
- [ ] b16最佳single与b2最佳single/pair在V分别选择；不读T/E效果决定。
- [ ] J集合可不嵌套；新增和丢失分别报告。
- [ ] 每个成功题一个canonical目标，不按成功次数加权。
- [ ] 空J返回父模型并保留计入；单个原子样本复用时保留统计相关性。
- [ ] 同协议两次用无偏pass@2；不同协议概率乘积只用于独立stream。

## C. SFT初始化、损失与恢复

- [ ] parent LoRA准确恢复后只开放原96模块；base/vision冻结，dropout0。
- [ ] fresh AdamW全臂一致；resume恢复本轮状态而非再次fresh。
- [ ] solver-J/self-J相同targets与配置时同一训练view；alias还检查parent/repeat/replay/optimizer/模板。
- [ ] 标准token mask、target首token shift、唯一EOS、PAD=EOS情形均正确。
- [ ] 序列先按completion长度平均；focus12/replay4各序列系数1/16。
- [ ] REPLAY_ONLY仅4样本仍保留.25总权重；无focus反向图。
- [ ] microbatch累积与定义目标梯度相等；全部累积后clip/step一次。
- [ ] 第1次更新lr=base/8，第8次base；恢复index不重复或漏步。
- [ ] 非法长输入不截断；不跨样本packing或遗留cache。
- [ ] CPU因果mask测试通过；真实Qwen单例/batch、未来后缀不影响前token、梯度非零等桥接通过。
- [ ] 技术bridge所做更新不进入正式训练；正式臂从parent重新开始。

## D. R0参照

- [ ] R0_RESET32使用新AdamW及8步warmup，标签明确不是原历史动量连续训练。
- [ ] 当前策略在线rollout，不能使用冻结teacher作behavior。
- [ ] 优势归一化和PPO token denominator与历史约定一致；B4/K8/epsilon1e-4。
- [ ] 全零优势加速保持零梯度与Adam/scheduler行为，不把grad=None当zero。
- [ ] 32步R0与256步SFT仅作上下文对照，不作同预算因果胜负。

## E. 评价、证据和时间

- [ ] 主终点256固定；V64/128诊断不改变各臂超参数或评价时刻。
- [ ] E包含384基础题，每题8次O0；image guard不混入主高分。
- [ ] PRE_ZERO16在parent的E响应上预先定义，只作解释，不提供训练输入。
- [ ] 0/n与n/n保留Wilson区间；不同checkpoint样本不混成单个策略IID。
- [ ] paired scene bootstrap保持所有parent/repeat及协议一同配对；两源轨迹的限制明确。
- [ ] 所有阴性、不利和空集结果保留；没有自动延长、加题或搜索赢家。
- [ ] 实际生成数、更新数、target exposure、cache与alias节省可复算。
- [ ] 恢复只认完整提交，不复制失败attempt；完整样本和权重路径都留存。

## F. 本包已执行的参考检查

在本包根目录运行：

```bash
cd reference
python -m unittest -v test_contracts
```

它们只使用确定性数组、固定数学例子与本轮配置。真实模型、真实新数据、服务器恢复、SFT因果性和显存仍需要上述实际验收。
