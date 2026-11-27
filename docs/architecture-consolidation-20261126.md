# 第二轮架构收敛：启动、目标与任务生命周期

本轮以 `0b9a0d3` 为行为基线，在本地实现八项审查意见。保留成熟的 TrackingController、FollowPolicy、VerticalSupervisor 和 motion/transport 事务；生产默认仍为中心瞄准、B 高度跟随、80% 巡航及 20% 固定左移 4 秒。所有验证使用本机、模拟串口或无设备 ARM 容器。

## 八项修改与验收位置

| 问题 | 最终实现 | 主要验证 |
| --- | --- | --- |
| F1 启动清理错误丢失 | `runtime.preparation.StartupService` 统一准备；`StartupResult` 明确 ready、stop_confirmed、error、stop_error。停止未确认写入账本及启动结果，恢复预览通过 `--startup-result` 接收，HealthRegistry 锁存 | 双重异常、清理失败不得返回成功、下一入口拒绝恢复；原有 PTY 归位顺序 |
| F2 两个启动账本 | 共同默认 `default_startup_state()`；PowerCycleStore 持有新旧路径锁，将旧账本未确认写入/暂停/停止错误保守迁入，保留原文件和迁移证据 | 跨入口未确认写入不重发；旧入口互斥；显式新周期保留迁移标记 |
| F3 网页与任务目标分离 | 任务发布 task_id / target_id / frame_id / timestamp 快照。状态和实际预览叠加层读取同一任务源；失去任务目标时不替换为预览选中的另一桶 | 双目标、遮挡、过期、侧向阶段交接；实际 preview_loop 绘框验证 |
| F4 两套观察生命周期 | `perception.observer.TargetObserver` 拥有读帧、校正、身份会话、恢复与退出；StereoObserver 为同一实现的交接适配。FrameBundle 是共同帧契约，get 返回同一种观察结构 | 自动初锁、遮挡恢复、共享原桶交接、两适配器共用 run/get 契约 |
| F5 计划与执行语义不同 | `TimedTranslate` 明示速度20%、4秒和“计时结束后禁用ACK”完成依据；b/2 改为 diagnostic_half_left_mm，仅供诊断和原有入场范围检查 | 计划/执行字段一致、原有 PTY 动作顺序、C 已伸出时不重复侧移 |
| F6 嵌套特征线程退出不确定 | request_stop / await_stopped 返回 CleanupResult；stop_and_drain 枚举嵌套身份工作，先停车后清理，超时保留错误并阻止交接 | 受控阻塞线程、停机顺序与清理超时；合成测试夹具同样关闭所属会话 |
| F7 主流程与模式交织 | CLI 在 options，生效配置在 profile，阶段转换在 stages，停稳后交接与动作在 completion；forward 负责追踪阶段组合 | 命名 profile、非法跨阶段跳转拒绝、180条原决策轨迹及任务回归 |
| F8 时序敏感与入口检查缺口 | PTY 检查真实命令减速包络、最终禁用及顺序；固定步长策略测试精确收敛4%。production_entries 清单同时覆盖 pyproject 正式入口与现场脚本 | 清单与正式入口完全匹配；现场脚本不得实现登记命令或底层写帧 |

## 人工维护时的唯一入口

- 改 A/B 控制：`tracking/controller.py`、`tracking/policy.py`；使用原180条行为轨迹对照。
- 改停止、ACK、命令编号：`motion/` 和 `transport/`。
- 改初始化/归位流程：`runtime/preparation.py`；工具脚本仅调用该服务。确认零位与物理回零继续区分。
- 改账本/启动周期：`runtime/startup.py`。不要删除旧账本解除未确认登记限制。
- 改追踪身份或恢复：`perception/session.py`；观察任务统一从 `perception/observer.py` 组合。
- 改生产参数：`resources/mission_profiles.json`。`production-center` 是生产默认，`diagnostic` 保留低速诊断，`experimental-a-priority` 显式启用 A 优先出发模式；光学瞄准区域仍由机构 reference 决定。
- 改任务阶段顺序：`missions/stages.py`、`missions/forward.py`、`missions/completion.py`。
- 改侧移计时策略：`missions/timed_translate.py`；几何投影保留在 post_geometry，不把估算位移写成实测值。
- 新增正式命令：同步 `resources/production_entries.json`，以契约检查约束入口。

发布清单携带 profile 和生产入口元数据；每轮任务另写 `mission_profile.json` 和 `task_lifecycle.jsonl`，记录实际参数和阶段。字段覆盖以运行时记录为准，允许显式命令行覆盖，但拒绝 A-only 与 A/B profile 的冲突。

## 显示与执行边界

任务快照是只读展示数据，控制器不从 JSON 快照读取控制输入。原始观测和实时串口仍直接进入原门控与事务。

任务期间，相机前向 Z 只有在原生选框与任务目标当前帧一致时才显示；身份、帧或时效不能确认时显示空值。神经辅助测距的“光学中点到桶表面距离”继续保留独立语义，不能冒充前向 Z。视频帧率仍由画面输出独立统计。

停止未确认会阻止 start、retry 和重启确认入口；普通重连不清除锁存。不可中断的 OpenCV 特征调用超过清理预算时，返回明确超时并保留故障，当前实现不承诺强行杀死 Python 线程。

## 本地验证

ARM 全量回归曾检出合成用例之间的线程交叉污染：前一用例的特征线程进入后一用例的 mock 范围。测试夹具现通过 addCleanup 关闭所属会话，目标恢复夹具还断言 CleanupResult.stopped。原有“不调用慢搜索”断言保留；恢复相关27项组合复测通过。失败轮日志独立保留，不用重跑成功覆盖失败证据。


执行证据保存于 `outputs/architecture_refactor_20261126/`，包括故障契约、全量回归、wheel、ARM 编译和无设备回放。ARM 容器无硬件设备、禁用网络；回放验证标定缺失、摘要不符、坏模型和 HTTP 外壳故障降级，未执行 NPU 推理或机构动作。

本地结果：macOS首轮409项无失败、8项环境跳过；ARM最终整轮414项，407通过、7项环境跳过；最终启动/所有权/清理契约16项通过，其中新增的账本权限分支在整轮收集后补测，合计覆盖415项。恢复隔离组合27项通过，原时序敏感的快速靠近串口用例连续3次通过；Ruff及147个文件格式检查通过。完整ARM编译、4个无设备回放场景和wheel独立导入通过，wheel与78个源码/资源文件逐项一致。

当前结论：控制计算与底层协议保持不变，重复状态和任务边界已收敛。核心证据为原行为轨迹、PTY 串口序列、故障契约和完整 ARM 构建。下一项有价值的验证是有人值守的板端集成验收，重点确认跨阶段目标显示一致、停止故障可见及当前中心/B/80%/4秒组合。
