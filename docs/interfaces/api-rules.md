# 视觉输出与后续 STM32 对接

网页模式只监听板端 `127.0.0.1:8788`，通过 SSH 转发访问。`/` 为预览页，`/stream.mjpg` 为最新 JPEG 流，`/frame.jpg` 为单帧，`/status` 为 JSON。HDMI 模式不启动 HTTP。

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
