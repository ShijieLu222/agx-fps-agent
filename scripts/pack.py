"""Build the single-file FPS submission; never upload the repository itself."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def pack(source=ROOT / "agent.py", output=ROOT / "dist/agx-fps-agent.zip"):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Output must not overwrite the source")
    data = source.read_bytes()
    if len(data) > 256 * 1024:
        raise ValueError("agent.py exceeds the platform's 256 KiB source limit")
    compile(data.decode("utf-8"), str(source), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("agent.py", date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(info, data, compresslevel=9)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "agent.py")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/agx-fps-agent.zip")
    args = parser.parse_args()
    try:
        digest = pack(args.source, args.output)
    except (OSError, ValueError, SyntaxError) as error:
        parser.error(str(error))
    print(f"Bundle: {args.output.resolve()}\nSHA256: {digest}\nContents: agent.py only")


if __name__ == "__main__":
    main()
