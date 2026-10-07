from __future__ import annotations

"""Costruisce e installa il kernel nativo `mars_core_py`.

Task 1 del piano di migrazione Rust. Esiste per una ragione precisa: la sequenza
corretta su questa macchina non e' quella che si trova nei tutorial di maturin.

- ``~/.cargo/bin`` non e' nel ``PATH`` delle shell non interattive, quindi
  ``cargo`` va risolto per percorso assoluto;
- ``maturin develop`` riporta "Installed" anche quando non installa
  nell'interprete corrente (senza virtualenv attivo l'installazione editabile
  finisce altrove e ``import mars_core_py`` fallisce con ``ModuleNotFoundError``);
  la sequenza affidabile e' ``maturin build`` seguito da ``pip install`` del wheel;
- su questa rete schannel non raggiunge il servizio di revoca dei certificati, per
  cui ogni accesso a crates.io fallisce senza ``CARGO_HTTP_CHECK_REVOKE=false``
  (impostato anche in ``rust/.cargo/config.toml``).

Uso::

    python scripts/build_rust.py             # build release + install
    python scripts/build_rust.py --check     # fmt, clippy -D warnings, test
    python scripts/build_rust.py --debug     # build non ottimizzata
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_ROOT = REPO_ROOT / "rust"
BINDING_DIR = RUST_ROOT / "mars_core_py"
DIST_DIR = RUST_ROOT / "dist"


def cargo_bin() -> str:
    """Percorso di ``cargo``, cercando prima nel ``PATH`` e poi in ``~/.cargo/bin``."""
    found = shutil.which("cargo")
    if found:
        return found
    fallback = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
    if fallback.exists():
        return str(fallback)
    raise SystemExit(
        "cargo non trovato: ne' nel PATH ne' in ~/.cargo/bin. "
        "Installare il toolchain con rustup prima di eseguire questo script."
    )


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    # `rust/.cargo/config.toml` copre le invocazioni fatte dentro `rust/`, ma
    # maturin puo' avviare cargo con un'altra directory di lavoro: la variabile
    # d'ambiente garantisce lo stesso comportamento in entrambi i casi.
    env.setdefault("CARGO_HTTP_CHECK_REVOKE", "false")
    cargo_dir = str(Path(cargo_bin()).parent)
    if cargo_dir not in env.get("PATH", ""):
        env["PATH"] = cargo_dir + os.pathsep + env.get("PATH", "")
    return env


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}   (in {cwd})", flush=True)
    result = subprocess.run(command, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise SystemExit(f"comando fallito ({result.returncode}): {printable}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build del kernel nativo mars_core_py.")
    parser.add_argument(
        "--check", action="store_true",
        help="esegue fmt, clippy con -D warnings e i test Rust, senza installare",
    )
    parser.add_argument("--debug", action="store_true", help="build non ottimizzata")
    parser.add_argument(
        "--vendor", action="store_true",
        help="scarica le dipendenze in rust/vendor per build successive offline",
    )
    args = parser.parse_args()

    cargo = cargo_bin()
    env = build_env()

    if args.check:
        run([cargo, "fmt", "--all", "--", "--check"], RUST_ROOT, env)
        run(
            [cargo, "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
            RUST_ROOT, env,
        )
        run([cargo, "test", "--workspace"], RUST_ROOT, env)
        print("\ncontrolli Rust superati.")
        return 0

    if args.vendor:
        run([cargo, "vendor", "vendor"], RUST_ROOT, env)
        print(
            "\nDipendenze scaricate in rust/vendor. Non e' versionata: aggiungerebbe "
            "decine di MB di sorgenti di terze parti a un repository di tesi. La "
            "riproducibilita' delle VERSIONI e' gia' garantita da Cargo.lock e dal "
            "pin del toolchain; questo comando aggiunge solo l'indipendenza dalla rete, "
            "ricreabile su richiesta."
        )
        return 0

    profile = [] if args.debug else ["--release"]
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DIST_DIR.glob("*.whl"):
        stale.unlink()

    run(
        [sys.executable, "-m", "maturin", "build", *profile, "--out", str(DIST_DIR)],
        BINDING_DIR, env,
    )

    wheels = sorted(DIST_DIR.glob("*.whl"))
    if not wheels:
        raise SystemExit("maturin non ha prodotto alcun wheel in rust/dist")

    run(
        [
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-index", "--no-deps", str(wheels[-1]),
        ],
        REPO_ROOT, env,
    )

    # Verifica in un processo NUOVO: nel processo corrente il modulo potrebbe
    # essere gia' caricato da una versione precedente e l'import riuscirebbe
    # anche se l'installazione appena fatta fosse rotta.
    check = subprocess.run(
        [
            sys.executable, "-c",
            "import mars_core_py; print(mars_core_py.version())",
        ],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    if check.returncode != 0:
        raise SystemExit(f"il modulo nativo non e' importabile:\n{check.stderr}")
    print(f"\nkernel nativo installato e importabile: {check.stdout.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
