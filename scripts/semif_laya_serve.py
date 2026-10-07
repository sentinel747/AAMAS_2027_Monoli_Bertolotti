"""Serve un checkpoint Laya LOCALE (es. il distillato da Jev) sul protocollo /v1/systemone.

`laya.serve` carica solo i checkpoint pubblicati; qui si costruisce lo stesso
server FastAPI di `laya.serve.create_app` ma con un Router a cui e' agganciato il
checkpoint della cartella indicata, sotto tutti e tre i nomi che il Router
conosce: nessuna richiesta puo' finire su un checkpoint di serie.

Gira nel venv di Laya (fastapi, uvicorn, torch), non in quello del simulatore:

    C:/tmp/laya_venv/Scripts/python.exe scripts/semif_laya_serve.py \
        --checkpoint runs/jev_semif_experiments/labelled_20260923/laya_distilled --port 8010

Poi `run_governor_experiment.py --semantic-provider laya --jev-endpoint http://127.0.0.1:8010`.
"""
from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="cartella del checkpoint Laya")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default=None, help="cuda / cpu (predefinito: cuda se c'e')")
    args = parser.parse_args(argv)

    import torch
    import uvicorn
    from laya.agent import Agent
    from laya.router import Router
    from laya.serve import create_app

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(args.checkpoint, device=device)
    router = Router(device=device, auto_task_detection=False)
    for name in ("english", "multilingual", "typed-decisions"):
        router.attach(name, agent)
    print(f"[laya-serve] checkpoint {args.checkpoint} su {device}, porta {args.port}", flush=True)
    uvicorn.run(create_app(router), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
