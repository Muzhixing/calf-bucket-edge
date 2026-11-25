# ADR 0001：标准包布局与控制能力所有权

状态：已采纳，软件实现与回归验证完成（2026-11-26）。

## 问题

行进任务最初只调用A控制并固定B0，遗漏成熟追踪中的B能力。后续抽取共享函数仍留下两套控制编排；串口事务、probe导入与日期目录拼装使更改难以传播和复现。

## 决策

使用Python标准src布局和pyproject.toml。按职责划分bucket_robot包；使用普通构造器注入和typing.Protocol描述设备边界。维持现有C++计算进程与Python控制，不引入服务框架、事件总线或依赖注入容器。

追踪组件统一目标变化、居中、边界与A/B调度；任务组件仅决定阶段与底盘策略。运动层管理机构/底盘命令约束；串口层管理独占、帧输出与确认。设备状态、目标观察和控制结果在接口处显式传递。

测试移至tests，现场工具至tools，实验至experiments；生产代码不能反向导入它们。旧入口保留小型兼容转发以便逐步迁移，核心实现在包内仅有一份。

## 取舍

只移动目录会保留重复行为；全面重写会同时改变已调好的控制与故障策略。因此采用渐进提取，并以原测试和行为对照证明迁移。不同任务的目标身份限制、ACK窗口、测距语义与NPU运行库隔离继续明确表达。

## 依据

- PyPA src布局：https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- PyPA pyproject：https://packaging.python.org/en/latest/specifications/pyproject-toml/
- Python结构化接口：https://typing.python.org/en/latest/reference/protocols.html
