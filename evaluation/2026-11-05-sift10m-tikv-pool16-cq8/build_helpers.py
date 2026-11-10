import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Build truth/count helpers with the SPTAGTest toolchain.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entry = next(item for item in json.loads((build / "compile_commands.json").read_text())
                 if item["file"].endswith("/TiKVIOTest.cpp"))
    compile_command = shlex.split(entry["command"])
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "SPTAGTest"], text=True)
    link = shlex.split(commands.splitlines()[-1])
    start = link.index(compile_command[0])
    end = link.index("&&", start) if "&&" in link[start:] else len(link)
    link = [item for item in link[start:end] if not item.endswith(".o")]
    for name in ("check_count", "exact_truth"):
        obj = output / (name + ".o")
        command = list(compile_command)
        command[command.index("-o") + 1] = str(obj)
        command[command.index("-c") + 1] = str(Path(__file__).parent / (name + ".cpp"))
        subprocess.run(command, cwd=entry["directory"], check=True)
        command = list(link)
        command.insert(1, str(obj))
        command[command.index("-o") + 1] = str(output / name)
        subprocess.run(command, cwd=build, check=True)
        obj.unlink()


if __name__ == "__main__":
    main()
