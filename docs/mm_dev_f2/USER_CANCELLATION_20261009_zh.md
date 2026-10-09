# MM-DEV F2 用户取消及结果删除

2026-10-09，用户明确请求：“停止这次的实验并删除结果”。最终状态为
`USER_CANCELLED_RESULTS_DELETED`。不得沿用历史授权恢复、重提或重建已删除的实验。

先取消依赖后继194849，再取消控制器194848和GPU作业194850，防止控制器继续派生作业。
三者均于16:14:44（新加坡时间）记为`CANCELLED`；GPU作业经历集群300秒KillWait，
待其完全退出队列后才开始服务器删除。额外KILL请求返回Invalid job id，未据此宣称
退出；最终使用实际队列和Slurm终态重新核验。

删除范围仅限这次MM-DEV F2：

- `/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009`
- `/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos`
- `/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos_cachefix`
- 独立工作树`artifacts/`下八个`mm_dev_f2_*_20261009`运行、下载和结果副本目录。
- `/private/tmp/mmdev-f2-review-gates-ulg25c79`及`mmdev_f2_visual_review_metadata.json`。

服务器共删除16,110个文件、768,962,449逻辑字节；本地副本2,477个文件，临时复核75个文件。
每个目标根先核对规范路径、非符号链接及逐文件SHA256，再复核清单未变，最后按白名单
删除。服务器规范路径及`/projects/varunssd/`别名均确认不存在；29个服务器相邻目录
和7个本地相邻目录身份不变。没有跟随符号链接删除共享资源。

未匹配到本次实验的Codex自动化；其他实验作业193374、193375保持运行。
旧MM-CORE数据、模型缓存、运行环境、Git源码及原始输入ZIP保留。Git中的实现/测试
和历史技术记录不等于仍保留模型输出或checkpoint；本文覆盖其历史继续执行指令。

不含实验payload的操作凭据存放于：

- 本地：`/Users/louis/Documents/ChatGPT/dissertation-ntu/artifacts/mm_dev_f2_cancellation_20261009`
- 服务器：`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_cancel_receipts_20261009`

记录包含取消回执、删除白名单、逐文件删除事件、受保护对象身份与独立删除核验。
服务器回执包已取回本地`SERVER_STOP_DELETE_RECEIPTS.tar.gz`，SHA256为
`c757572d6fb1387c8ab71fb865cb7785a8a809288ec6261c866c7c34110f1877`。
本地核验通过：8个安全唯一归档文件、gzip CRC/读至EOF、清单全部SHA256，以及
16,110条删除事件与白名单逐项匹配。此次仅更新取消记录，未修改训练代码或重跑测试/模型。
这是用户取消和清理完成；不是GRPO恢复通过、科学矩阵完成或科学汇报交付。
