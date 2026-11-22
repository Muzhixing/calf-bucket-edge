"""Offline tool. Does not import the runtime, connect serial, or activate a profile.

Input JSON: source, calibration_sha256, stationary flags, scope, commands and
observations. Both arrays contain [monotonic_timestamp, value]; observations are
the image x coordinate of a confirmed stationary bucket, commands are A software
telemetry. Scope specifies measured B_mm and distance_m intervals, and C_mm=10.
"""
import argparse,hashlib,json
from pathlib import Path
from response import fit_response,ResponseProfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();source=Path(args.input);destination=Path(args.output)
    if destination.exists():raise ValueError('refuse_to_overwrite_profile')
    data=json.loads(source.read_text());fit=fit_response(data)
    result=dict(schema=1,source=data.get('source','simulation'),calibration_sha256=data.get('calibration_sha256'),
                scope=data.get('scope'),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),**fit)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    # Rejected/simulated results remain diagnostic artifacts, never loadable as
    # a hardware profile. A good fit alone does not prove hardware acceptance.
    if result['source']=='field_log' and fit['validation']['accepted']:
        ResponseProfile.load(destination,result['calibration_sha256'])
    print(json.dumps({'output':str(destination),'source':result['source'],**fit},indent=2))
    return 0 if fit['validation']['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
