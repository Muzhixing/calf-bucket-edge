# 视觉输出与 STM32 追踪控制

网页与 HDMI 模式均只监听板端 `127.0.0.1:8788`，通过 SSH 转发访问。`/` 为追踪控制页，`/stream.mjpg` 为追踪叠加 JPEG 流，`/frame.jpg` 为单帧，`/status` 为 JSON。原生后端位于后一端口（默认 8789），`/capture.jpg` 的 `X-Frame-Id`、`X-Observation` 和原始左图原子对应；不对外开放。

`/status.tracking` 包含 `mode`（auto/manual）、`active`（操作员跟随意图）、`vision_state`（searching/tracking/reacquiring/unavailable）、`reacquired_count`、`target_id`、`target`（box/offset_px/quality/timestamp/method；边缘路径另有partial_visible）、`reason`、`motion_available`、`limits` 和 `arm`（phase/telemetry/timestamp）。控制图像有效年龄为 0.4 秒，独立于原测距 1.5 秒有效期。`arm.phase=holding` 表示已居中持续观察，`reacquiring` 表示已暂停机构等待原目标重获，`stop_unconfirmed` 表示尚未收到禁用确认。

控制 POST `/api/start`（`{"chassis_isolated":true}`）、`stop`、`auto`、`select`（`frame_id` 与 `box`）、`heartbeat` 需要本次页面的 `X-Control-Token`。框选在实时图像上进行，松开时 GET `/snapshot.jpg` 取得最新图及响应头帧号，再立即提交框坐标；截图有效 0.4 秒，初始化后光流衔接当前帧。选择目标时先停止机械臂，桶和车无需静止。旧`--step-control`模式要求页面heartbeat，超过3秒请求禁用；默认自动连续模式独立于页面生命周期。短暂目标丢失时 `active` 可保持 true，但 `target=null` 且机构暂停；自动模式通过当前图像/检测确认后续跟，旧身份超时后自动重新获取，不将旧位置当成当前目标；手动旧模式保留超时停用。相机故障或用户停止不会自动恢复机构。没有 REST 自动复位零位或扩大行程接口。

| 字段 | 含义 |
|---|---|
| `frame_id`、`frame_age_s` | 当前来源帧编号、距发布的秒数；超过 1.5 秒或帧号停止推进时视为离线。 |
| `detected`、`selected` | 是否检出桶与所选框，坐标位于 1280×720 校正左目。 |
| `distance_m`、`raw_distance_m` | 有效平滑 Z 深度、本次新测原始深度；无值为 null。 |
| `measurement_status` | `new` 新测、`held` 沿用、`invalid` 无有效距离。 |
| `distance_source_frame`、`distance_age_ms` | 有效距离的源帧及年龄；不可把沿用值当作当前帧新测。 |
| `sampling_roi` | 来源帧的实际采样矩形 `[x1,y1,x2,y2]`；不等同于桶口。 |
| `distance_reason`、`attempt_source`、`discarded` | 最近一次测距结果原因、尝试源帧及是否因过时/换目标被丢弃。 |
| `guidance.offset_px` | 桶框中心减画面中心的 `[dx,dy]`，右、下为正；丢失为 null。 |
| `guidance.estimated_offset_mm` | 根据有效深度和校正焦距估算横纵偏移；无有效深度为 null。 |
| `guidance.aligned_xy` | 两轴均在容差内：横向 ±51.2 px，纵向 ±28.8 px，含边界。 |
| `guidance.state` | `no_target`、`adjust`、`aligned_wait_depth`、`ready_dry_run`。 |
| `guidance.reference_depth_mm` | 本次连续对齐期间已取得新距离后，提供桶身前向深度，单位 mm。 |
| `guidance.cycle_id`、`guidance.feed_preview_started` | 一次连续对齐的预演周期；首次具备条件时仅一帧标记 true，轮询方按周期去重。 |

同步读取同一条记录中的位置、距离、状态与源帧。丢失、更换目标或距离超过 1.5 秒会清空旧值。当前容差是视觉联调值，毫米偏差基于同一深度平面估计。

这些字段是观察量和预演条件，不是电机脉冲或运行时间。实际接入 STM32 前需确认出水口零位、轴方向、每毫米脉冲数/编码器比例、行程限位、桶前壁到投放点偏移，并由控制器管理任务开始、完成回执和防重复投喂。
