# F′评测图片路由故障与原池恢复

2026-10-23 19:15 UTC重新连通服务器后，确认7张GPU正在执行，已有五条16步训练完成。旧任务保持评测触发 `PermissionError: Image path escapes run root`。新TOPK回答已生成；错误发生在首个旧任务保持组读取图片时、该组模型调用之前。

原因是准备程序给旧TEST_ID和ChartQA写入原实验DATA目录的绝对图片路径，而原生运行时只允许本次运行根DATA内的图片。路径边界检查本身正确。修复准备程序，复制注册图片到本次DATA/retained_images，改用相对路径，保留原image_sha256及其他字段；原生加载继续进行其已有的图片身份检查，不另加哈希扫描或GPU烟测。

对已冻结的当前运行，仅以 `sr_f4.c_pilot_images` 做一次有记录的文件路由修订：保存旧IMAGE_MANIFEST及FROZEN_INPUTS，记录逐图前后路径和修复代码版本，然后同步修订这两份清单中的图片路径。图片内容、题表、提示、随机种子、回答预算、FROZEN.json、CONFIG.json、checkpoint身份和全部已生成输出保持不变。锁住任务池，并确认已领取的子进程已读完清单，再切换；保留中断恢复记录。科学冻结的初始代码版本保持原值，修复版本独立记录，不伪装成初始冻结前已修复。

只恢复迁移时登记的FAILED或RUNNING评测attempt。后续只有该确切attempt以相同路径错误失败、有匹配时间段的技术事件、无COMPLETE回执且lease释放时，才恢复为PENDING。原attempt、错误日志、回答全部保留；新attempt再出错不自动重试。正在运行并已读入旧清单的评测可能稍后遇到同样故障，可再次执行同一恢复命令处理这些已登记attempt。训练不终止、不重跑；现有worker每任务启动新子进程，未来任务读取新路径即可，不必热替换运行中的模型代码。

恢复命令（修复代码commit并push、部署后运行）：

```bash
PYTHONPATH=/path/to/fixed-code/src python -m sr_f4.c_pilot_images \
  --root /projects/_ssd/varunssd/louis-ssvc/sr_f4_c_fprime_pilot_20261023 \
  --revision FIX_COMMIT
```

完成的回答按原group_id、qid、draw和seed复用。生产验证关注恢复分片是否超过已完成的64个新TOPK组，并开始产生原固定OLD_TEST_ID回答；不另开测试生成、不新增Slurm作业。其他用户作业与配额不作修改。
