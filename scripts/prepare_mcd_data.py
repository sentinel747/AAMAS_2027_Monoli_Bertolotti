from __future__ import annotations

import argparse
import os
import stat
import tarfile
from pathlib import Path


ARCHIVES = {
    "MCD_6.1.tar.gz": "base MCD distribution",
    "MCD6.1.tar.gz": "base MCD distribution",
    "MCD_pr 6.1.tar.gz": "base MCD distribution",
    "clim_minEUV.tar.gz": "minimum solar EUV climatology",
    "clim_maxEUV.tar.gz": "maximum solar EUV climatology",
    "cold.tar.gz": "low dust and minimum solar EUV",
    "warm.tar.gz": "high dust and maximum solar EUV",
    "strm.tar.gz": "dust storm scenario",
}


def _readable_mode(path: Path, is_dir: bool) -> int:
    if os.name == "nt":
        return 0o755 if is_dir else 0o644
    current = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0
    base = current or (0o755 if is_dir else 0o644)
    base |= stat.S_IRUSR | stat.S_IWUSR
    if is_dir:
        base |= stat.S_IXUSR
    if base & stat.S_IRUSR:
        base |= stat.S_IRGRP | stat.S_IROTH
    if base & stat.S_IWUSR:
        base |= stat.S_IWGRP
    if is_dir and base & stat.S_IXUSR:
        base |= stat.S_IXGRP | stat.S_IXOTH
    return base


def _ensure_readable(path: Path, is_dir: bool) -> None:
    try:
        os.chmod(path, _readable_mode(path, is_dir))
    except OSError:
        # Best effort only. Verification later reports unreadable paths.
        return


def _known_archive_paths(source: Path) -> list[tuple[Path, bool]]:
    paths: list[tuple[Path, bool]] = []
    for archive_name in ARCHIVES:
        archive_path = source / archive_name
        if not archive_path.exists():
            continue
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                paths.append((Path(member.name), member.isdir()))
    return paths


def repair_permissions(target: Path, source: Path | None = None) -> list[str]:
    target = target.resolve()
    logs: list[str] = []
    if not target.exists():
        return [f"target does not exist: {target}"]

    _ensure_readable(target, True)
    if source is not None and source.exists():
        for relative_path, is_dir in _known_archive_paths(source.resolve()):
            current = target
            for parent in relative_path.parents:
                if str(parent) == ".":
                    continue
                _ensure_readable(target / parent, True)
            current = target / relative_path
            _ensure_readable(current, is_dir)
    for root, dirs, files in os.walk(target):
        root_path = Path(root)
        _ensure_readable(root_path, True)
        for name in dirs:
            _ensure_readable(root_path / name, True)
        for name in files:
            _ensure_readable(root_path / name, False)
    logs.append(f"repaired permissions under {target}")
    return logs


def verify_layout(target: Path) -> list[str]:
    target = target.resolve()
    checks = {
        "climatology": target / "MCD_6.1" / "data" / "clim_aveEUV" / "clim_01_me.nc",
        "clim_minEUV": target / "clim_minEUV" / "clim_01_thermo_min_me.nc",
        "clim_maxEUV": target / "clim_maxEUV" / "clim_01_thermo_max_me.nc",
        "cold": target / "cold" / "cold_01_me.nc",
        "warm": target / "warm" / "warm_01_me.nc",
        "strm": target / "strm" / "strm_07_me.nc",
    }
    logs: list[str] = []
    for scenario, sample in checks.items():
        try:
            readable = sample.exists() and os.access(sample, os.R_OK)
        except PermissionError:
            readable = False
        if readable:
            logs.append(f"verified {scenario}: {sample}")
        else:
            logs.append(f"missing_or_unreadable {scenario}: {sample}")
    return logs


def _safe_extract(archive_path: Path, target: Path) -> int:
    target = target.resolve()
    extracted = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if target != destination and target not in destination.parents:
                raise ValueError(f"unsafe archive member path: {member.name}")
        for member in archive.getmembers():
            archive.extract(member, target, filter="data")
            _ensure_readable((target / member.name).resolve(), member.isdir())
            extracted += 1
    return extracted


def prepare(source: Path, target: Path, repair_only: bool = False, verify: bool = True) -> list[str]:
    source = source.resolve()
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    _ensure_readable(target, True)

    logs: list[str] = []
    if repair_only:
        logs.extend(repair_permissions(target, source))
    else:
        for archive_name, label in ARCHIVES.items():
            archive_path = source / archive_name
            if not archive_path.exists():
                continue
            count = _safe_extract(archive_path, target)
            logs.append(f"extracted {archive_name} ({label}): {count} entries")
        logs.extend(repair_permissions(target, source))

    if not logs:
        logs.append(f"no MCD archives found in {source}")
    if verify:
        logs.extend(verify_layout(target))
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract or repair MCD 6.1 and scenario archives into a local data folder.")
    parser.add_argument("--source", default=".", help="Folder containing MCD_6.1.tar.gz and add-on scenario tarballs.")
    parser.add_argument("--target", default="data/mcd", help="Destination folder ignored by git.")
    parser.add_argument("--repair-only", action="store_true", help="Do not extract archives again; only normalize permissions under the target folder.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip post-run layout verification.")
    args = parser.parse_args()

    for line in prepare(Path(args.source), Path(args.target), repair_only=args.repair_only, verify=not args.skip_verify):
        print(line)
    print("Use this path in the GUI Climate data path field:")
    print(Path(args.target).resolve())


if __name__ == "__main__":
    main()
