"""Print the environment and verify a real MPS operation. Does not load YOLO."""
import json
import sys

from mac_runtime import environment, require_mps


def main():
    report = environment()
    try:
        require_mps()
        report['mps_smoke_test'] = 'passed'
    except RuntimeError as exc:
        report['mps_smoke_test'] = 'failed'
        report['error'] = str(exc)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['mps_smoke_test'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
