"""Stopped task handoff and post-actions, using the mature motion adapters."""

import json
from bucket_robot.missions.lifecycle import park_fault
from bucket_robot.missions.stages import Stage


def complete_mission(args, result, near_measurement, post_plan, observer, handoff, port, stages):
    p = port
    if near_measurement is not None and not result["error"] and not result["stop_error"]:
        result["approach"] = dict(
            mode="distance_first",
            reason="target_distance_reached",
            error=None,
            stop_error=None,
            final_state=result["final_state"],
            last_measurement=near_measurement,
        )
        if args.post_approach_plan:
            from bucket_robot.missions.post_approach import pending_extension_plan

            result["post_approach"] = post_plan or pending_extension_plan(result["approach"])
            (args.out / "post_approach.json").write_text(
                json.dumps(result["post_approach"], indent=2)
            )
        (args.out / "task_status.json").write_text(
            json.dumps(
                dict(
                    phase="awaiting_execution_confirmation"
                    if args.post_approach_plan
                    else "complete",
                    reason="target_distance_reached",
                    disabled_confirmed=True,
                ),
                indent=2,
            )
        )
        (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    if (
        args.approach_model
        and not args.observe_only
        and not args.center_only
        and not result["error"]
        and not result["stop_error"]
        and result.get("reason") == "left_reference_reached"
    ):
        stages.transition(Stage.APPROACH)
        try:
            stages.check()
            latest = observer.snapshot()
            if (
                latest[0]
                and latest[0].get("target") is not None
                and latest[1] is not None
                and not latest[2]
            ):
                obs, image, error = latest
            elif handoff is not None:
                obs, image, error = handoff
            else:
                raise RuntimeError("invalid_approach_handoff")
            if error or image is None or obs is None or obs.get("target") is None:
                raise RuntimeError("invalid_approach_handoff")
            import cv2
            from bucket_robot.missions.approach import run_approach

            seed = args.out / "left_handoff.png"
            if not cv2.imwrite(str(seed), image):
                raise RuntimeError("handoff_image_write_failed")
            if args.recover_loss:
                (args.out / "task_status.json").write_text(
                    json.dumps(
                        dict(phase="left_approach", status_file="approach/task_status.json"),
                        indent=2,
                    )
                )
            if hasattr(observer.thread, "is_alive") and observer.thread.is_alive():
                raise RuntimeError("tracking_observer_handoff_not_stopped")
            result["approach"] = run_approach(
                out=args.out / "approach",
                reference=args.reference,
                calibration=args.approach_calibration,
                model=args.approach_model,
                seed_image=seed,
                seed_box=obs["target"].get("raw_box", obs["target"]["box"]),
                native_port=args.native_port,
                execute=True,
                bearing_tracker=args.bearing_tracker,
                recover_loss=args.recover_loss,
                seed_timestamp=obs["target"]["timestamp"],
                verified_surface_band=args.approach_surface_band,
                continuous=args.distance_first,
                existing_port=p if args.distance_first else None,
                target_session=getattr(observer, "target_session", None)
                if args.recover_loss
                else None,
                align_upper_after_stop=args.a_start_only,
                continuous_speed=20 if args.cruise_speed > 20 else 8,
            )
            result["operator_stopped"] = result["approach"].get("operator_stopped", False)
            if result["approach"]["error"] or result["approach"]["stop_error"]:
                result["error"] = "approach_failed"
            if args.post_approach_plan and not result["error"] and not result["operator_stopped"]:
                from bucket_robot.missions.post_approach import pending_extension_plan

                result["post_approach"] = pending_extension_plan(result["approach"])
                (args.out / "post_approach.json").write_text(
                    json.dumps(result["post_approach"], indent=2)
                )
            if args.recover_loss:
                final = result["approach"].get("final_state") or {}
                (args.out / "task_status.json").write_text(
                    json.dumps(
                        dict(
                            phase="stopped"
                            if result["operator_stopped"]
                            else "fault_paused"
                            if result["error"]
                            else "awaiting_execution_confirmation"
                            if args.post_approach_plan
                            else "complete",
                            reason=result["approach"].get("reason") or result["error"],
                            disabled_confirmed=bool(final)
                            and not final.get("enabled", True)
                            and not result["approach"]["stop_error"],
                        ),
                        indent=2,
                    )
                )
        except BaseException as exc:
            result["error"] = repr(exc)
            result["operator_stopped"] = isinstance(exc, KeyboardInterrupt)
        finally:
            if p is not None and p.fd is not None:
                result["stop_error"] = p.shutdown() or result["stop_error"]
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if (
        args.execute_post_geometry
        and (result.get("reason") == "post_left_aligned" or result.get("approach"))
        and not result.get("operator_stopped")
        and not result["error"]
        and not result["stop_error"]
    ):
        from bucket_robot.missions.post_execute import execute_post

        stages.transition(Stage.POST_ACTION)
        stages.check()
        result["post_execution"] = execute_post(
            post_plan or result["post_approach"], args.reference, args.out
        )
        result["error"] = result["post_execution"]["error"]
        result["stop_error"] = result["post_execution"]["stop_error"]
        result["final_state"] = result["post_execution"].get("final_state")
        result["operator_stopped"] = result["post_execution"].get("operator_stopped", False)
        (args.out / "summary.json").write_text(json.dumps(result, indent=2))
        print("POST_EXECUTION " + json.dumps(result["post_execution"]), flush=True)
    stages.transition(Stage.FAULT if result["error"] or result["stop_error"] else Stage.COMPLETE)
    if (
        args.recover_loss
        and not args.observe_only
        and (result["error"] or result["stop_error"])
        and not result.get("operator_stopped")
    ):
        park_fault(args.out, result)
    return 0 if not result["error"] and not result["stop_error"] else 2
