"""Build upstream host tests with mocks and export real SystemControl_Status.

Usage: python3 verify_firmware.py /path/to/MCU-slave-device
No board connection. Does not edit the upstream checkout.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from bucket_robot.transport.telemetry import PROTOCOL_COMMIT

source = Path(sys.argv[1]).resolve()
actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
if actual != PROTOCOL_COMMIT:
    raise SystemExit("Firmware source commit mismatch: " + actual)
project = source / "new_project_f407/F407_HAL_Template"
with tempfile.TemporaryDirectory(prefix="f407-oracle-") as temporary:
    build = Path(temporary)
    for path in (project / "tests/mocks").glob("*.h"):
        shutil.copy2(path, build / path.name)
    for name in ("UpperPacket", "AxisControl", "SystemControl", "StepperPulse"):
        for ext in ("c", "h"):
            shutil.copy2(project / "Hardware" / (name + "." + ext), build)
    shutil.copy2(project / "Hardware/HWT101_Parser.h", build)
    original = (project / "tests/test_control.c").read_text()
    marker = ' puts("PASS '
    if original.count(marker) != 1:
        raise SystemExit("Expected host-test export point not found")
    export = ' printf("ORACLE=");for(i=0;i<20;i++)printf("%02X",out[i]);puts("");\n'
    (build / "test_control.c").write_text(original.replace(marker, export + marker))
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(build),
            *map(str, sorted(build.glob("*.c"))),
            "-lm",
            "-o",
            str(build / "test"),
        ],
        check=True,
    )
    result = subprocess.check_output([str(build / "test")], text=True)
    print(result, end="")
    hex_frame = next(
        line.removeprefix("ORACLE=") for line in result.splitlines() if line.startswith("ORACLE=")
    )
fixture = {
    "source_repository": "https://github.com/Muzhixing/MCU-slave-device.git",
    "commit": actual,
    "mode": "upstream C with host mocks; no hardware",
    "hex": hex_frame,
    "source_hashes": {
        str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [project / "Hardware/SystemControl.c", project / "tests/test_control.c"]
    },
}
Path(__file__).with_name("firmware_fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
