# 接入实际独立核验的PILOT_V3

上游233920已实际完成独立修复核验，10份原件一次交付并核引用，状态CONFIRMATION_GATES_PASS，全部执行权限仍false。原6e的9216回答、1e的360审核和24条数值轨迹不重跑，原1e UNKNOWN与历史失败不变。来源、结论与限制详见[CLOSURE_REPAIR_EXECUTION](CLOSURE_REPAIR_EXECUTION_zh.md)。

当前新增 `historical_closure` 作为新生产者的完整只读来源桥接。旧 `historical_confirmation`、旧763实际部署及旧a803操作器保留。新桥接分别记录原6e生产者、1e数值/审核执行者、1879修复核验者，消费实际新登记与封存；不能给旧UNKNOWN重新标记PASS。隔离子进程只导入这三个已冻结来源，冷核验现有支持、审核及数值闭包；只在证据未变化时复用当前进程已验证结果。

新PILOT生产源码将独立提交/部署；新操作目录 `scripts/sr_f6/pilot_v3_closure_operations` 和私有 `.local/pilot_v3_closure_20261029` 只接受该实际已提交源码、完整来源差异及实际修复交付。原S0数据/父来源仍按原件绑定，旧S0 FAIL不翻转；复用已经核实的模型和adapter内容记录，不重新扫描权重。

实施顺序保持：服务器实际CPU回归与共享文件系统检查→真实source delta/S0来源→隔离核验来源绑定→固定384 DEV_PILOT和96 DEV_EVAL公开v3输入覆盖→原三父tensor和九路计划→原24数值共享实现对应→独立PILOT_V3准入→两独立单GPU worker作业计划。GPU外层提交器须在实际READY_FOR_REGISTERED_PILOT_ONLY之后完成和检查。

科学预算和门完全不变：9×12步，唯一b0/U11 step4到12恢复，3父加9端点的DEV_EVAL，U11非零信用组/root/家族及至少两个父模型的更新后信用延续门。不得新增效果胜出门，不改题、seed、臂、lambda或步数。FORMAL/TEST仍需新生产来源和消费者。

新桥接与操作器已实施，完整独立复核后无剩余critical项。修复了子进程继承外层缓存目录、渲染操作器与选定生产清单不一致的风险；原来源读取、编号、数值计算、门、数据和预算保持。37项真实隔离子进程专项、180项来源/修复联合回归通过；另62项注册/准入/共享数值实现/完整模拟训练恢复检查通过。模拟结果不构成科学通过。

本记录的源码提交节点尚无新部署、PILOT登记、CPU准备意图、覆盖层、准入、GPU提交或模型调用。实际生产revision必须取完成提交并核实推送后的值；后续执行事实追加到此文件及私有续接材料，不能用文档更新revision冒充实际生产源。原部署763工程检查不代表新接入的服务器验证。

## 11:03 UTC新生产实际部署与唯一准备提交

新生产源码 `b7e7b067f237b360b1dfb2c3075d3f14be7c281f` 已commit/push并用ls-remote核实；旧763/6e/1e/1879部署均保持。新680项完整执行源码清单及174项S0差异由实际Git归档生成；新旧操作器31项检查通过，同一部署归档的140项隔离CPU测试通过，Ruff/格式/Shell/diff通过。真实数据、日志、实际审核证据均未上传Git。

独立部署 `sr_f6_code_b7e7b06`（758归档成员、2,268,780 bytes；SHA `3151d97ac5de5220aacc3fcc1d92f221da638287cb26a7a4440d607f4ac209bd`），服务器14实际模块冷导入成功、0模型调用。新操作器激活前核对完整10份修复交付、完整原S0/三父引用以及渲染源码与所选生产清单字节相同；旧a803没有激活。

11:02:43 UTC，在原233723/233849/233873/233920均完成离队、无新旧准备意图或阶段目录、teacher全用户实际1GPU/varun024的1槽后，唯一提交0GPU `233935/LH-SRF6-PilotV3-ClosurePrep`（4CPU12G）。11:03:42实际RUNNING/59秒，服务器native回归returncode0，PILOT_V3来源登记BOUND_NOT_AUTHORIZED，隔离历史核验一个attempt仍进行中、无failure。尚无确认binding/overlay/先导准入或GPU模型调用。

不要重复deploy、生成请求、submit、register或复跑旧确认。当前只读唯一233935及阶段回执；终态完整READY_FOR_REGISTERED_PILOT_ONLY后才能实现并审查外层GPU提交器。技术故障保留原件，不自动重签/补抽；科学FAIL/INCONCLUSIVE按原合同停止。实际准备回执与续接见私有`.local/pilot_v3_closure_20261029/CONTINUE_zh.md`。后续文档提交不改变实际生产b7e7b06。

## 11:27 UTC实际先导准入完成和原件交付

233935已COMPLETED0:0，耗时12分32秒。服务器140项native回归通过（16.82秒）；来源、独立确认重建、固定384+96公开覆盖、完整九路注册、原模型/环境及共享数值实现对应全部核验完成。11:15:15原结果READY_FOR_REGISTERED_PILOT_ONLY，生成/训练/开发评估仅在原先导范围开放，FORMAL/TEST仍false；0新回答/0更新/0GPU提交。

phase identity为`d9f061d1cf08e14c485083bbed421a412a29868fd8305ad4c265fb6404562284`，pilot spec为`4936b29e3a897334c1c4e67aad436bf46bdea9a105354d0155484d2acc20994c`，admission为`b65831ad6ec57c829a69a9ad0a87a6aaf6a4a173e2d921cf953462ca7519b2a6`，jobs plan为`56102cd5b10d93bde1996527e39fa1ec30943fc2618cab2838f03d5fd550d4f4`。共享数值对应仍明确new_trainer_numerical_pass=false，未扩大原24轨迹的测量范围。

11:27:30已完成一次实际原件交付：50文件7,230,616bytes，111项内部引用核验，临时cache/tmp明确省略；旧原始依赖没有重复下载。首次传输在读取文件前因Slurm已purge作业实时条目而停止，改用全用户当前队列确认离队后唯一成功传输；随后本机曾误把空stderr.log解析为JSON，使用已收到字节完成核验，没有再次访问服务器。两项传输/解析技术记录完整保留，不改变实际准入结果。

当前仅开始外层Round A操作器实施，独立于已冻结b7生产源码和准入。先导实际GPU尚未提交。实际后续记录见[PILOT_ROUND_A_EXECUTION_zh.md](PILOT_ROUND_A_EXECUTION_zh.md)。
