"""Legacy mission entry forwards to the shared always-on application."""
from bucket_robot.runtime.application import mission_main as main
from bucket_robot.missions.coordinator import command as command

if __name__ == "__main__":
    raise SystemExit(main())
