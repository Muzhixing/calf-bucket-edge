"""Build the existing mission worker command; runtime.application owns its lifecycle."""

import sys


def command(layout, out, port, budget):
    return [
        sys.executable,
        "-m",
        "bucket_robot",
        "forward",
        "--recover-loss",
        "--track-b",
        "--post-approach-plan",
        "--reference",
        str(layout.config / "arm_reference.json"),
        "--left-reference",
        str(layout.config / "left_view_reference.json"),
        "--calibration",
        str(layout.assets / "calibration.yml"),
        "--approach-model",
        str(layout.assets / "fast_acv.rknn"),
        "--approach-calibration",
        str(layout.assets / "calibration.json"),
        "--approach-surface-band",
        ".35",
        ".78",
        ".65",
        ".95",
        "--out",
        str(out / "mission"),
        "--auto-target",
        "--expect-a",
        "73",
        "--expect-b",
        "0",
        "--seconds",
        str(budget),
        "--until-left",
        "--bearing-tracker",
        "--native-port",
        str(port),
    ]
