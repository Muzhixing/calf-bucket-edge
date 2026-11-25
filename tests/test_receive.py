"""Offline tests only: synthetic bytes are never sent to a physical port."""

import json
from pathlib import Path
import random
import unittest

from bucket_robot.transport.telemetry import Parser, decode, summary


def frame(**fields):
    # Independent fixture builder, deliberately separate from the receiver.
    values = [0xB5, 1, 255, 0, 0, 0x3E, 0xD2, 4, 0x29, 9, 0xF9, 0x0D, 0xEA, 0, 90, 0, 0, 7, 0, 0xB6]
    for offset, value in fields.items():
        values[int(offset)] = value
    values[18] = 0
    for value in values[1:18]:
        values[18] ^= value
    return bytes(values)


class ReceiveTests(unittest.TestCase):
    def test_fields_endianness_units_and_flags(self):
        item = decode(frame())
        self.assertFalse(item["enabled"])
        self.assertTrue(item["lift_zero_confirmed"])
        self.assertTrue(item["slide_zero_confirmed"])
        self.assertEqual(item["lift_position_mm"], 123.4)
        self.assertEqual(item["slide_position_mm"], 234.5)
        self.assertEqual(item["yaw_deg"], 357.7)
        self.assertEqual(item["ultrasound_mm"], 234)
        self.assertEqual(item["command_count_mod256"], 255)
        self.assertEqual(item["uart_error_count_mod256"], 7)

    def test_all_single_splits_and_bytewise(self):
        data = frame()
        for split in range(1, 20):
            parser = Parser()
            self.assertEqual(parser.feed(data[:split]), [])
            self.assertEqual(len(parser.feed(data[split:])), 1)
            self.assertEqual(parser.buffer, b"")
        parser, output = Parser(), []
        for byte in data:
            output.extend(parser.feed(bytes([byte])))
        self.assertEqual(len(output), 1)

    def test_concatenated_same_counter_and_wrap(self):
        parser = Parser()
        data = frame() + frame() + frame(**{"2": 0})
        self.assertEqual([x["command_count_mod256"] for x in parser.feed(data)], [255, 255, 0])

    def test_invalid_checksum_version_and_tail(self):
        good = frame()
        bad_sum = bytearray(good)
        bad_sum[6] ^= 1
        bad_tail = bytearray(good)
        bad_tail[19] = 0
        parser = Parser()
        result = parser.feed(bytes(bad_sum) + frame(**{"1": 2}) + bytes(bad_tail) + good)
        self.assertEqual(len(result), 1)
        self.assertEqual(parser.stats["bad_checksum"], 1)
        self.assertEqual(parser.stats["bad_version"], 1)
        self.assertEqual(parser.stats["bad_tail"], 1)

    def test_noise_truncation_embedded_headers_and_random_chunking(self):
        good = frame(**{"6": 0xB5, "7": 0xB6})
        data = b"garbage" + frame()[:8] + b"\x11\x22" + good * 30
        parser, output, rng, offset = Parser(), [], random.Random(407), 0
        while offset < len(data):
            count = rng.randint(1, 53)
            output.extend(parser.feed(data[offset : offset + count]))
            offset += count
        self.assertEqual(len(output), 30)
        self.assertTrue(all(x["hex"] == good.hex(" ").upper() for x in output))
        self.assertLess(len(parser.buffer), 20)

    def test_old_10byte_protocol_is_not_accepted(self):
        old = bytes.fromhex("B5 01 00 00 FF 00 00 57 00 B6")
        parser = Parser()
        self.assertEqual(parser.feed(old * 10), [])
        self.assertEqual(len(parser.feed(frame())), 1)

    def test_invalid_sentinels_and_validity_flags(self):
        for data in (
            frame(**{"10": 255, "11": 255, "12": 255, "13": 255}),
            frame(**{"5": 0}),
            frame(**{"5": 0x18}),
        ):
            item = decode(data)
            self.assertIsNone(item["yaw_deg"])
        self.assertIsNone(decode(frame(**{"5": 0}))["ultrasound_mm"])
        self.assertIsNone(decode(frame(**{"12": 255, "13": 255}))["ultrasound_mm"])

    def test_timing_uses_host_times_not_command_counter(self):
        parser = Parser()
        records = parser.feed(frame() * 3)
        for i, item in enumerate(records):
            item["monotonic_ns"] = i * 250_000_000
        result = summary(records, parser)
        self.assertEqual(result["interval_ms_median"], 250)
        self.assertEqual(result["command_counts"], [255])
        self.assertEqual(result["enabled_frames"], 0)
        self.assertIsNone(summary([], Parser())["interval_ms_median"])

    def test_real_firmware_oracle_if_available(self):
        fixture = Path(__file__).parent / "fixtures/firmware_fixture.json"
        if not fixture.exists():
            self.skipTest("Generate firmware fixture using verify_firmware.py first")
        oracle = json.loads(fixture.read_text())
        item = decode(bytes.fromhex(oracle["hex"]))
        self.assertEqual(item["command_count_mod256"], 9)
        self.assertEqual(item["result"], 4)
        self.assertEqual(item["state"], 4)
        self.assertFalse(item["enabled"])
        self.assertEqual(item["system_fault"], 2)
        self.assertEqual(item["motor_fault"], 1)
        self.assertEqual(item["ultrasound_mm"], 234)
        self.assertEqual(item["uart_error_count_mod256"], 3)


if __name__ == "__main__":
    unittest.main()
