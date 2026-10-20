"""Inspect, plan and install a pinned reference environment on the Linux server."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "environments/hstu-reference-requirements.txt"
MANIFEST = ROOT / "configs/hstu-reference-upstream.json"


def run(command, **kwargs):
    """Run an argument list without shell interpolation."""
    return subprocess.run(command, check=True, text=True, **kwargs)


def digest(path):
    """Hash a small specification file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outside_project(path):
    """Prevent physical environments and upstream checkouts inside this project."""
    path = path.expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Environment and upstream paths must be outside the project")
    return path


def check_platform():
    """Guard the architecture and glibc required by the pinned wheel."""
    libc, version = platform.libc_ver()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This setup is for the Linux x86_64 server only")
    if libc != "glibc" or tuple(int(x) for x in version.split(".")[:2]) < (2, 28):
        raise RuntimeError(f"The wheel requires glibc >= 2.28; found {libc} {version}")


def conda_executable():
    """Find Conda without using the server's base Python for project execution."""
    candidate = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not candidate and Path("/opt/conda/bin/conda").is_file():
        candidate = "/opt/conda/bin/conda"
    if not candidate:
        raise RuntimeError("Conda executable not found")
    return candidate


def inspect():
    """Display current hardware and package state without installing anything."""
    print(json.dumps({"platform": platform.platform(), "machine": platform.machine(),
                      "libc": platform.libc_ver(), "python": sys.version,
                      "executable": sys.executable, "prefix": sys.prefix,
                      "nvcc": shutil.which("nvcc"), "sbatch": shutil.which("sbatch")}, indent=2), flush=True)
    run([conda_executable(), "--version"])
    run([sys.executable, "-m", "pip", "--version"])
    if shutil.which("nvidia-smi"):
        run(["nvidia-smi"])
    run([sys.executable, "-c", "import importlib.metadata as m, json; "
         "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() "
         "if d.metadata['Name'].lower().replace('_','-') in "
         "{'torch','fbgemm-gpu','numpy','pandas','gin-config','tensorboard','absl-py','ipykernel'}}, indent=2))"])


def validate_report(report, prefix):
    """Reject a stale dependency plan before applying it."""
    state = json.loads((report / "state.json").read_text())
    if state != {"requirements_sha256": digest(SPEC), "prefix": str(prefix)}:
        raise RuntimeError("Plan belongs to a different prefix or specification; run plan again")
    planned = json.loads((report / "pip-plan.json").read_text())
    constraints = "".join(sorted(
        f"{entry['metadata']['name']}=={entry['metadata']['version']}\n"
        for entry in planned["install"]
    ))
    if not constraints or (report / "resolved-constraints.txt").read_text() != constraints:
        raise RuntimeError("Resolved constraints do not match the saved dependency plan")


def prepare(action, prefix, report):
    """Resolve first, then install with all dependency versions constrained."""
    check_platform()
    python = prefix / "bin/python"
    if action == "plan":
        report.mkdir(parents=True, exist_ok=True)
        (report / "state.json").unlink(missing_ok=True)
        if not python.is_file():
            if prefix.exists():
                raise RuntimeError("Target path exists without Python; choose a fresh prefix")
            run([conda_executable(), "env", "create", "--prefix", str(prefix),
                 "--file", str(ROOT / "environments/hstu-reference.yml")])
    version = run([str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
                  capture_output=True).stdout.strip()
    if version != "3.11.16":
        raise RuntimeError(f"Expected Python 3.11.16, found {version}")
    if action == "plan":
        (report / "before-freeze.txt").write_text(
            run([str(python), "-m", "pip", "freeze", "--all"], capture_output=True).stdout)
        run([str(python), "-m", "pip", "install", "--dry-run", "--ignore-installed",
             "--report", str(report / "pip-plan.json"), "-r", str(SPEC)])
        planned = json.loads((report / "pip-plan.json").read_text())
        constraints = "".join(sorted(
            f"{entry['metadata']['name']}=={entry['metadata']['version']}\n"
            for entry in planned["install"]
        ))
        (report / "resolved-constraints.txt").write_text(constraints)
        (report / "state.json").write_text(json.dumps({
            "requirements_sha256": digest(SPEC), "prefix": str(prefix)}, indent=2))
        print(f"Dependency plan saved: {report}")
        print("No project libraries installed. Review resolved-constraints.txt before install.")
    else:
        validate_report(report, prefix)
        # PEP 440 can treat a local build as satisfying the public release version.
        # Replace only FBGEMM explicitly, without reinstalling torch or its dependencies.
        installed = run([str(python), "-c", "import importlib.metadata as m\n"
                         "try: print(m.version('fbgemm-gpu'))\n"
                         "except m.PackageNotFoundError: print('')"], capture_output=True).stdout.strip()
        expected = json.loads(MANIFEST.read_text())["expected_packages"]["fbgemm-gpu"]
        if installed and installed != expected:
            requirement = next(line for line in SPEC.read_text().splitlines()
                               if line.startswith("fbgemm-gpu @ "))
            run([str(python), "-m", "pip", "install", "--force-reinstall", "--no-deps",
                 "--report", str(report / "pip-fbgemm-replace.json"), requirement])
        run([str(python), "-m", "pip", "install", "--report", str(report / "pip-install.json"),
             "-r", str(SPEC), "-c", str(report / "resolved-constraints.txt")])
        run([str(python), "-m", "pip", "check"])
        (report / "installed-freeze.txt").write_text(
            run([str(python), "-m", "pip", "freeze", "--all"], capture_output=True).stdout)
        run([str(python), "-m", "ipykernel", "install", "--user",
             "--name", "agenticrec-hstu-reference", "--display-name", "AgenticRec (HSTU reference)"])
        print(f"Reference environment ready: {prefix}")


def source(path):
    """Fetch exactly the selected upstream commit without installing its extras."""
    spec = json.loads(MANIFEST.read_text())
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", spec["repository"], str(path)])
    # A clone can succeed while checkout fails, for example on shared storage.
    # Resume only when no index or user files exist; never reset existing work.
    if ((path / ".git").is_dir() and not (path / ".git/index").exists()
            and not any(entry.name != ".git" for entry in path.iterdir())):
        remote = run(["git", "-C", str(path), "remote", "get-url", "origin"],
                     capture_output=True).stdout.strip()
        if remote != spec["repository"]:
            raise RuntimeError("Uninitialized checkout has a different origin; not modifying it")
        run(["git", "-C", str(path), "checkout", "--detach", spec["revision"]])
    revision = run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    dirty = run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True).stdout.strip()
    if revision != spec["revision"] or dirty:
        raise RuntimeError("Existing upstream checkout is modified or at another revision; not overwriting")
    for name, expected in spec["files_sha256"].items():
        if digest(path / name) != expected:
            raise RuntimeError(f"Upstream source mismatch: {name}")
    print(f"Pinned upstream verified: {revision}")


def main():
    """Keep dependency planning, installation and source checkout explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "plan", "install", "source"))
    parser.add_argument("--prefix", type=Path, default=Path.home() / "conda-envs/agenticrec-hstu-reference")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "runs/hstu-reference-setup")
    parser.add_argument("--upstream-dir", type=Path, default=Path.home() / "upstream/generative-recommenders-reference")
    args = parser.parse_args()
    if args.action != "inspect" and sys.version_info[:2] != (3, 11):
        raise RuntimeError("Activate the project's Python 3.11 baseline environment before setup")
    if args.action == "inspect":
        inspect()
    elif args.action == "source":
        source(outside_project(args.upstream_dir))
    else:
        prefix = outside_project(args.prefix)
        if prefix == Path(sys.prefix).resolve():
            raise RuntimeError("Run setup from the existing baseline environment, not the target prefix")
        prepare(args.action, prefix, args.report_dir.resolve())


if __name__ == "__main__":
    main()
