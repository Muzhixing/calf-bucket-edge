"""Protocol telemetry positions in command degrees and millimetres."""


def pose(state):
    return [state["servo_software_deg"], state["lift_position_mm"], state["slide_position_mm"]]
