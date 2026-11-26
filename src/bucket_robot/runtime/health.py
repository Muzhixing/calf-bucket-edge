"""Thread-safe component health, capability gates and latched motion faults."""

from dataclasses import asdict, dataclass
import threading
import time


@dataclass
class ComponentStatus:
    state: str = "unknown"
    code: str = "NOT_OBSERVED"
    message: str = "尚无有效状态"
    first_seen: float = 0
    last_seen: float = 0
    count: int = 0
    recovery: str = "等待有效数据"
    ttl: float = 0


class HealthRegistry:
    dependencies = {
        "preview": ("camera", "preview"),
        "detection": ("camera", "calibration", "detector"),
        "visual_tracking": ("camera", "calibration", "detector", "tracker"),
        "arm_tracking": (
            "camera",
            "calibration",
            "detector",
            "tracker",
            "motion_config",
            "journal",
            "serial",
            "arm",
        ),
        "mission": (
            "camera",
            "calibration",
            "detector",
            "motion_config",
            "journal",
            "serial",
            "arm",
            "chassis",
            "approach",
        ),
    }
    names = {
        "camera": "摄像头",
        "preview": "画面输出",
        "detector": "目标检测",
        "calibration": "相机标定",
        "tracker": "视觉追踪",
        "serial": "STM32通信",
        "arm": "A/B/C机构",
        "chassis": "四轮底盘",
        "motion_config": "机构配置",
        "approach": "靠近测距",
        "ranging": "实时测距",
        "journal": "日志",
        "backend": "视觉后端",
        "application": "应用",
    }

    def __init__(self, clock=time.monotonic, motion_enabled=True):
        self.clock, self.motion_enabled = clock, motion_enabled
        self.lock = threading.RLock()
        self.components = {name: ComponentStatus() for name in self.names}
        self.motion_latch = None
        self.boot_pose = [73, 0, 0]

    def report(self, component, state, code, message, *, recovery="自动重新检查", ttl=0):
        with self.lock:
            now = self.clock()
            old = self.components.get(component, ComponentStatus())
            first = old.first_seen if (old.code, old.state) == (code, state) else now
            self.components[component] = ComponentStatus(
                state,
                code,
                message,
                first,
                now,
                old.count + 1 if (old.code, old.state) == (code, state) else 1,
                recovery,
                ttl,
            )

    def ready(self, component, message="正常", ttl=0):
        self.report(component, "ready", "OK", message, ttl=ttl, recovery="无需处理")

    def fault(self, code, message):
        with self.lock:
            if self.motion_latch is None or code == "STOP_UNCONFIRMED":
                self.motion_latch = {"code": code, "message": message, "since": self.clock()}

    def clear_motion_latch(self):
        """Only an explicit operator resume may release a recovered fault."""
        with self.lock:
            self.motion_latch = None

    def telemetry(self, frames):
        if not frames:
            return
        s = frames[-1]
        self.ready("serial", "已收到有效控制器回传", ttl=0.7)
        ready = (
            s.get("lift_zero_confirmed")
            and s.get("slide_zero_confirmed")
            and s.get("angle_valid")
            and s.get("heading_reference_established")
            and not s.get("system_fault")
            and not s.get("motor_fault")
        )
        initial_boot = (
            not s.get("enabled")
            and s.get("command_count_mod256") == 0
            and not s.get("lift_zero_confirmed")
            and not s.get("slide_zero_confirmed")
            and [s.get("servo_software_deg"), s.get("lift_position_mm"), s.get("slide_position_mm")]
            == self.boot_pose
        )
        if s.get("motor_fault") or s.get("system_fault") and not initial_boot:
            self.fault("CONTROLLER_FAULT", "控制器报告故障，运动已锁定")
        for component in ("arm", "chassis"):
            if ready:
                self.ready(component, "控制器回传就绪；未提供独立电机在线证明", ttl=0.7)
            else:
                self.report(
                    component,
                    "unavailable",
                    "CONTROLLER_NOT_READY",
                    "控制器未就绪，请检查零位、航向及故障回传",
                    ttl=0.7,
                    recovery="处理原因后明确恢复运动",
                )

    def disconnected(self, message="未收到有效控制器回传"):
        with self.lock:
            previous = self.components["serial"]
            if previous.state == "ready":
                self.fault("SERIAL_LOST", "通信中断，运动暂停；停止状态需确认")
        self.report(
            "serial", "unavailable", "STM32_OFFLINE", message, recovery="自动重连；运动需明确恢复"
        )
        for name in ("arm", "chassis"):
            self.report(
                name,
                "unknown",
                "DEVICE_STATE_UNKNOWN",
                "控制器未连接，电机状态无法确认",
                recovery="等待控制器有效回传",
            )

    def snapshot(self):
        with self.lock:
            now = self.clock()
            components = {}
            for name, item in self.components.items():
                row = asdict(item)
                row["name"] = self.names.get(name, name)
                if item.ttl and now - item.last_seen > item.ttl:
                    row.update(state="stale", code="DATA_STALE", message="状态数据已过期")
                    if name == "serial" and item.state == "ready":
                        self.fault("SERIAL_STALE", "控制器回传过期，运动暂停")
                components[name] = row
            capabilities = {}
            for name, dependencies in self.dependencies.items():
                blocked = [key for key in dependencies if components[key]["state"] != "ready"]
                motion = name in ("arm_tracking", "mission")
                reasons = [components[key]["message"] for key in blocked]
                if motion and not self.motion_enabled:
                    reasons.append("当前为只读观察模式")
                if motion and self.motion_latch:
                    reasons.append(self.motion_latch["message"])
                capabilities[name] = {
                    "available": not reasons,
                    "blocked_by": blocked,
                    "reasons": reasons,
                }
            return {
                "state": "running"
                if all(
                    capabilities[k]["available"] for k in ("preview", "detection", "arm_tracking")
                )
                else "degraded",
                "components": components,
                "capabilities": capabilities,
                "motion_latch": self.motion_latch,
                "motion_enabled": self.motion_enabled,
            }

    def allows(self, capability):
        return self.snapshot()["capabilities"][capability]["available"]
