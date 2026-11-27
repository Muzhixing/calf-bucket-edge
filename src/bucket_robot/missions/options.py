"""CLI adaptation into explicit mission options; no hardware or control logic."""

import argparse
from pathlib import Path
from bucket_robot.missions.profile import PROFILE_CATALOG


def parse_options(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--left-reference", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--seed-image")
    ap.add_argument("--seed-box", nargs=4, type=int)
    ap.add_argument("--native-port", type=int, default=8899)
    ap.add_argument(
        "--auto-target",
        action="store_true",
        help="Acquire with original automatic tracker; retain that identity for this mission",
    )
    ap.add_argument("--expect-a", type=int, default=97)
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--observe-only", action="store_true")
    ap.add_argument(
        "--until-left",
        action="store_true",
        help="Continuous alignment; drive budget maximum 60 s legacy / 120 s recovery mission",
    )
    ap.add_argument(
        "--bearing-tracker",
        action="store_true",
        help="Require current associated detector evidence for visible bucket bearing, including vertically clipped views",
    )
    ap.add_argument(
        "--center-only",
        action="store_true",
        help="Verify A-only acquisition and stop before any chassis motion",
    )
    ap.add_argument(
        "--recover-loss",
        action="store_true",
        help="Keep task alive, ramp down on visual loss and resume only a verified same target",
    )
    axes = ap.add_mutually_exclusive_group()
    axes.add_argument(
        "--a-only",
        action="store_true",
        help="Explicit A-only diagnostic; ordinary missions use --track-b",
    )
    axes.add_argument(
        "--track-b",
        action="store_true",
        help="Pause the chassis and track vertical error with B before resuming",
    )
    ap.add_argument("--expect-b", type=int, default=0)
    ap.add_argument(
        "--post-approach-plan",
        action="store_true",
        help="Record forward alignment, timed left translation, B-top and C150 plan",
    )
    ap.add_argument(
        "--approach-surface-band",
        nargs=4,
        type=float,
        help="Scene-verified exterior wall region as relative box coordinates",
    )
    ap.add_argument(
        "--approach-model",
        type=Path,
        help="After confirmed left alignment, run bounded classical/neural left approach",
    )
    ap.add_argument(
        "--approach-calibration", type=Path, help="Hash-bound original JSON stereo calibration"
    )
    ap.add_argument(
        "--distance-first",
        action="store_true",
        help="Range from lock; stop near 20cm at any A angle",
    )
    ap.add_argument("--cruise-speed", type=int, default=8, choices=range(1, 81))
    ap.add_argument("--a-start-only", action="store_true")
    ap.add_argument("--execute-post-geometry", action="store_true")
    ap.add_argument(
        "--profile",
        choices=tuple(PROFILE_CATALOG["profiles"]),
        default="diagnostic",
    )
    selected, _ = ap.parse_known_args(argv)
    ap.set_defaults(**PROFILE_CATALOG["profiles"][selected.profile])
    args = ap.parse_args(argv)
    if args.a_only and args.track_b:
        ap.error("A-only diagnostic cannot use an A/B profile")
    if args.a_start_only and not args.track_b:
        ap.error("A-priority departure requires --track-b for parked upper alignment")
    if args.cruise_speed > 20 and not (args.distance_first or args.observe_only):
        ap.error("high cruise speed requires distance-first assisted ranging")
    if args.execute_post_geometry and not (args.distance_first and args.post_approach_plan):
        ap.error("post geometry execution requires distance-first and post-approach-plan")
    if not (args.track_b or args.a_only or args.observe_only or args.center_only):
        ap.error(
            "select --track-b for shared A/B tracking, or explicitly --a-only for a diagnostic"
        )
    if args.auto_target:
        if args.seed_image or args.seed_box:
            ap.error("automatic acquisition cannot use a manual seed")
    elif not args.seed_image or not args.seed_box:
        ap.error("provide --auto-target or both seed arguments")
    if args.track_b and not args.recover_loss:
        ap.error("--track-b requires --recover-loss")
    if args.distance_first and (
        not args.recover_loss or not args.approach_model or args.observe_only
    ):
        ap.error("distance-first requires recovering motion and assisted ranging")
    if args.post_approach_plan and not args.approach_model:
        ap.error("post-approach plan requires range approach")
    if bool(args.approach_model) != bool(args.approach_calibration):
        ap.error("approach model and calibration must be supplied together")
    return args
