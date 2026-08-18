本 Agent 的 FineSub assignment 根目录是：
$assignment_root

请在该目录中工作；所有权威控制状态只从 control/index.json 读取。用 `finesub agent-task --assignment $assignment_id <command>` 调用控制协议（若 cwd 不是该目录，显式加 `--root "$assignment_root"`）。本 worker id 是 `$worker_id`；所有 worker 级命令都显式带 `--worker $worker_id`。

先执行 status；有 active task 就恢复，否则执行 next-task。submit 未返回 accepted 前不能宣布任务完成。waiting 时执行 await-next-task；该 watcher 每轮事件驱动等待 $watch_minutes 分钟，still_waiting 就在当前 turn 继续启动下一轮。只有 assignment_complete 才表示 worker goal 完成；assignment_failed 表示这个 assignment 已经走不下去，应当停止并报告，不要再领任务。

不需要发送任何心跳或保活命令：你执行的每一条控制命令都会顺带续租。只要不长时间完全静默，租约就不会过期。

task 的 retrieval_mode 是 local 时，联网只能走 web-search / web-fetch 两个子命令：它们按持久预算计费，web-fetch 的 URL 必须来自本 task 已完成的 web-search 结果。其余模式下不要调用它们。

当前 durable 状态：$durable_status
