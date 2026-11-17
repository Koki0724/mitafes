"""Download the pinned YOLO11 weights, or verify existing files."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request


MODELS = {
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo11m-pose.pt": "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2",
    "yolo11s.pt": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
}
BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def verify(path, expected):
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}\nexpected {expected}\nactual   {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--check", action="store_true", help="Verify only; do not download")
    args = parser.parse_args()
    for name, expected in MODELS.items():
        destination = args.directory / name
        if destination.exists() or args.check:
            verify(destination, expected)
        else:
            args.directory.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {name} ...", flush=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=args.directory, suffix=".part", delete=False) as out:
                    temporary = Path(out.name)
                    with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
                        shutil.copyfileobj(response, out)
                verify(temporary, expected)
                # Avoid replacing an existing model, including one created during download.
                destination.hardlink_to(temporary)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        print(f"Verified {destination}", flush=True)


if __name__ == "__main__":
    main()
