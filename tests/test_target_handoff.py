"""A stage handoff reuses the live target session after its old worker stops."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from bucket_robot.perception.observer import StereoObserver


class TargetHandoffTests(unittest.TestCase):
    def test_stereo_stage_updates_same_session_without_selecting_again(self):
        manager = SimpleNamespace(state="tracking", generation=1, reacquired=0, reason="")
        observer = StereoObserver(
            8899,
            SimpleNamespace(maps=[(None, None)]),
            "unused",
            [400, 200, 650, 450],
            recover_loss=True,
            target_session=manager,
        )
        calls = []

        def update(image, metadata):
            calls.append(metadata["frame_id"])
            observer.end.set()
            return dict(timestamp=1, box=[400, 200, 650, 450])

        manager.update = update

        class Capture:
            def __init__(self, *args, **kwargs):
                pass

            def read(self):
                return np.zeros((720, 2560, 3), np.uint8), dict(frame_id=7, arrival=1)

        with (
            patch("bucket_robot.perception.capture.NativeCapture", Capture),
            patch("cv2.remap", side_effect=lambda image, *args: image),
            patch("cv2.imread", side_effect=AssertionError("must not re-seed from a file")),
        ):
            observer.run()
        self.assertIsNone(observer.error)
        self.assertIs(observer.target_session, manager)
        self.assertEqual(calls, [7])
        self.assertEqual(observer.get()["identity"], 1)
