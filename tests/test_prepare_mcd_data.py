import io
import tarfile

from scripts.prepare_mcd_data import prepare, repair_permissions, verify_layout


def _write_member(archive: tarfile.TarFile, name: str, payload: bytes = b"nc", mode: int = 0o000) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = mode
    archive.addfile(info, fileobj=io.BytesIO(payload))


def test_prepare_extracts_and_verifies_mcd_layout(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()

    with tarfile.open(source / "MCD_6.1.tar.gz", "w:gz") as archive:
        _write_member(archive, "MCD_6.1/data/clim_aveEUV/clim_01_me.nc")
    with tarfile.open(source / "warm.tar.gz", "w:gz") as archive:
        _write_member(archive, "warm/warm_01_me.nc")

    logs = prepare(source, target)

    assert (target / "MCD_6.1" / "data" / "clim_aveEUV" / "clim_01_me.nc").exists()
    assert (target / "warm" / "warm_01_me.nc").exists()
    assert any(line.startswith("verified climatology:") for line in logs)
    assert any(line.startswith("verified warm:") for line in logs)


def test_repair_permissions_keeps_existing_tree_verifiable(tmp_path):
    target = tmp_path / "target"
    sample = target / "warm" / "warm_01_me.nc"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"nc")

    logs = repair_permissions(target)
    verification = verify_layout(target)

    assert any("repaired permissions under" in line for line in logs)
    assert any(line.startswith("verified warm:") for line in verification)
