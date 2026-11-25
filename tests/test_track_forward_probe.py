import json, os, pty, select, tempfile, threading, time, tty, unittest
from pathlib import Path
from bucket_robot.missions.forward import ForwardGate, CombinedPort, Limits


class GateTests(unittest.TestCase):
    def t(self, now, dx=0):
        return {"timestamp": now, "offset_px": [dx, 0]}

    def test_center_before_forward_and_timeout(self):
        g = ForwardGate(5)
        self.assertEqual(g.evaluate(1, self.t(1), 1, 97, True), (0, None))
        self.assertEqual(g.evaluate(1.8, self.t(1.8), 1, 97, True), (8, None))
        self.assertEqual(g.evaluate(6.81, self.t(6.81), 1, 100, True), (0, "time_limit"))

    def test_ten_second_boundary(self):
        g = ForwardGate(10)
        g.evaluate(1, self.t(1), 1, 97, True)
        g.evaluate(2, self.t(2), 1, 97, True)
        self.assertEqual(g.evaluate(12, self.t(12), 1, 100, True), (0, "time_limit"))
        with self.assertRaises(ValueError):
            ForwardGate(10.1)

    def test_continuous_task_crosses_old_segment_boundary_and_stops_at_left(self):
        g = ForwardGate(60, until_left=True)
        g.evaluate(1, self.t(1), 1, 97, True)
        g.evaluate(2, self.t(2), 1, 97, True)
        self.assertEqual(g.evaluate(15, self.t(15), 1, 110, True), (8, None))
        self.assertEqual(g.evaluate(40, self.t(40), 1, 131, True), (0, "left_reference_reached"))

    def test_continuous_task_keeps_timeout_loss_and_identity_stops(self):
        g = ForwardGate(60, until_left=True)
        g.evaluate(1, self.t(1), 1, 97, True)
        g.evaluate(2, self.t(2), 1, 97, True)
        for target, identity in ((None, 1), (self.t(15), 2), (self.t(10), 1)):
            with self.assertRaises(RuntimeError):
                g.evaluate(15, target, identity, 110, True)
        self.assertEqual(g.evaluate(62, self.t(62), 1, 120, True), (0, "time_limit"))
        for seconds in (60.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ForwardGate(seconds, until_left=True)

    def test_image_center_alone_is_not_left(self):
        self.assertEqual(ForwardGate(5).evaluate(1, self.t(1), 1, 97, True), (0, None))
        self.assertEqual(
            ForwardGate(5).evaluate(1, self.t(1), 1, 132, True), (0, "left_reference_reached")
        )

    def test_loss_stale_swap_and_tracking_error(self):
        for kind in ("lost", "stale", "swap", "error"):
            g = ForwardGate(5)
            g.evaluate(1, self.t(1), 1, 97, True)
            g.evaluate(1.8, self.t(1.8), 1, 97, True)
            t = (
                None
                if kind == "lost"
                else self.t(1 if kind == "stale" else 2, 50 if kind == "error" else 0)
            )
            with self.assertRaises(RuntimeError):
                g.evaluate(2, t, 2 if kind == "swap" else 1, 99, True)

    def test_moving_A_does_not_start_drive(self):
        g = ForwardGate(5)
        for n in range(20):
            self.assertEqual(g.evaluate(n / 10, self.t(n / 10), 1, 97, False), (0, None))

    def test_left_limit_with_error_stops(self):
        self.assertEqual(
            ForwardGate(5).evaluate(1, self.t(1, -80), 1, 132, True), (0, "left_boundary_reached")
        )


class Sender:
    def __init__(self, port, serial_factory, command_repeats):
        self.fd = serial_factory(port, 115200)

    def _open_serial(self):
        pass

    def write_control_frame(self, frame, repeats):
        return self.fd.write(frame) == 10


class MCU:
    def __init__(self):
        self.m, self.s = pty.openpty()
        tty.setraw(self.s)
        self.path = os.ttyname(self.s)
        os.set_blocking(self.m, False)
        self.a = 97
        self.enabled = False
        self.vx = 0
        self.count = 0
        self.commands = []
        self.end = threading.Event()
        self.drop = False
        self.buf = b""
        self.yaw = 0
        self.uart_errors = 0
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def packet(self):
        f = bytearray(
            [
                181,
                1,
                self.count,
                0,
                3 if self.enabled and self.vx else 1 if self.enabled else 0,
                46 | int(self.enabled),
                0,
                0,
                0,
                0,
                0,
                0,
                255,
                255,
                self.a,
                0,
                0,
                0,
            ]
        )
        f[10:12] = int(self.yaw).to_bytes(2, "little")
        f[17] = self.uart_errors
        c = 0
        for v in f[1:18]:
            c ^= v
        return f + bytes([c, 182])

    def run(self):
        sent = 0
        while not self.end.wait(0.005):
            if select.select([self.m], [], [], 0)[0]:
                try:
                    self.buf += os.read(self.m, 4096)
                except OSError:
                    continue
            while len(self.buf) >= 10:
                f = self.buf[:10]
                self.buf = self.buf[10:]
                self.commands.append(f)
                self.count = (self.count + 1) % 256
                self.enabled = bool(f[8])
                self.vx = f[1]
                if self.enabled:
                    self.a = f[4]
            if not self.drop and time.monotonic() - sent > 0.04:
                os.write(self.m, self.packet())
                sent = time.monotonic()

    def close(self):
        self.end.set()
        self.thread.join()
        os.close(self.m)
        os.close(self.s)


class PortTests(unittest.TestCase):
    def setUp(self):
        self.m = MCU()
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "ref.json"
        path.write_text(
            json.dumps(
                {
                    "operating_envelope": {
                        "A_command_degrees": [73, 132],
                        "B_command_mm": [0, 96],
                        "C_trial_hold_mm": 0,
                    },
                    "control_policy": {"original_sender_module_on_rk3588": "unused"},
                }
            )
        )
        self.p = CombinedPort(
            Limits(path),
            lambda *args: None,
            port=self.m.path,
            sender_factory=Sender,
            check_holders=False,
        )
        self.p.rearm()

    def tearDown(self):
        self.p.shutdown()
        self.m.close()
        self.tmp.cleanup()

    def ack(self):
        until = time.monotonic() + 0.5
        while time.monotonic() < until:
            self.p.tick()
            if not self.p.pending:
                return
            time.sleep(0.01)
        self.fail("no ack")

    def test_speed_changes_with_unchanged_A_and_confirmed_stop(self):
        self.assertTrue(self.p.request(97, 0))
        self.ack()
        time.sleep(0.05)
        self.assertTrue(self.p.request(97, 8))
        self.ack()
        time.sleep(0.05)
        self.assertTrue(self.p.request(99, 8))
        self.ack()
        self.assertEqual(self.p.state["state"], 3)
        self.assertIsNone(self.p.shutdown())
        self.assertFalse(self.p.state["enabled"])
        self.assertEqual(self.m.commands[-1][1:3], bytes([0, 0]))
        self.assertEqual(self.m.commands[-1][8], 0)
        for f in self.m.commands:
            self.assertEqual(f[5:8], bytes([0, 0, 0]))

    def test_stale_serial_rejected_and_no_blind_retry(self):
        self.m.drop = True
        self.p.rx_at = time.monotonic() - 1
        n = len(self.m.commands)
        with self.assertRaises(RuntimeError):
            self.p.request(97, 8)
        time.sleep(0.04)
        self.assertEqual(len(self.m.commands), n)
        self.m.drop = False

    def test_new_A_and_speed_bounds(self):
        for a, v in ((133, 8), (97, 9), (103, 8)):
            with self.assertRaises(RuntimeError):
                self.p.request(a, v)

    def test_only_one_unacknowledged_frame(self):
        self.assertTrue(self.p.request(98, 8))
        self.assertFalse(self.p.request(99, 8))
        self.ack()

    def test_heading_fault_still_waits_for_confirmed_disable(self):
        self.assertTrue(self.p.request(97, 8))
        self.ack()
        self.m.yaw = 40
        with self.assertRaisesRegex(RuntimeError, "heading_drift"):
            until = time.monotonic() + 0.5
            while time.monotonic() < until:
                self.p.tick()
                time.sleep(0.01)
        self.assertIsNone(self.p.shutdown())
        self.assertFalse(self.p.state["enabled"])
        self.assertEqual(self.p.state["state"], 0)

    def test_uart_fault_still_waits_for_confirmed_disable(self):
        self.assertTrue(self.p.request(97, 8))
        self.ack()
        self.m.uart_errors = 1
        with self.assertRaisesRegex(RuntimeError, "new_uart_error"):
            until = time.monotonic() + 0.5
            while time.monotonic() < until:
                self.p.tick()
                time.sleep(0.01)
        self.assertIsNone(self.p.shutdown())
        self.assertFalse(self.p.state["enabled"])


if __name__ == "__main__":
    unittest.main()
