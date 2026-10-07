"""Distilla le decisioni di Jev in Laya (encoder System One open-weights).

Gira nell'ambiente Python di Laya (`laya`, `torch`), NON in quello del
simulatore: non importa nulla del repository. Legge le domande raccolte da
`semif_harvest_cases.py` e le distribuzioni di Jev prodotte da
`semif_label_cases.py`, addestra la sola testa di decisione di Laya con
l'encoder congelato (entra in 4 GB di GPU), e salva un checkpoint locale che
`laya.serve` carica con `LAYA_MODEL_PATH` / `Agent(<cartella>)`.

Metodo:
- stesso input di `Agent.system_one` (`build_sequence`, stessi marcatori);
- encoder calcolato una sola volta per caso, senza gradienti, in fp16;
- testa (type_emb, 2 strati transformer, scorer) addestrata con entropia
  incrociata sui bersagli morbidi di Jev;
- divisione train/validazione con seme fisso; si conserva l'epoca migliore sulla
  validazione;
- temperature d'inferenza del checkpoint a 1: i logit imparano direttamente le
  probabilita' del maestro.

    <laya_venv>/python scripts/semif_distill_laya.py \\
        --harvest runs/.../harvest.jsonl --teacher runs/.../jev.jsonl \\
        --out-dir runs/.../laya_distilled --epochs 8
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import ssl
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tls_windows_store() -> None:
    """Come src/llm/tls.py: negozio di Windows + CA del progetto, verifica attiva.

    Serve solo se i pesi non sono ancora in cache (download da Hugging Face
    dietro l'intercettazione HTTPS dell'antivirus).
    """
    bundle = REPO / "configs" / "ca_bundle.pem"
    originale = ssl.create_default_context

    def contesto(*args, **kwargs):
        ctx = originale(*args, **kwargs)
        ctx.load_default_certs()
        if bundle.exists():
            ctx.load_verify_locations(cafile=str(bundle))
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        return ctx

    ssl.create_default_context = contesto


def _question(row: dict) -> dict:
    return {
        "t": "choice",
        "ins": row["question"],
        "crit": {o["id"]: o["description"] for o in row["options"]},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--cases", type=Path,
                        default=REPO / "configs" / "experiments" / "jev_semif" / "agent_cases_v1.json")
    args = parser.parse_args(argv)
    _tls_windows_store()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from laya import DEFAULT_MODELS
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, collate_items

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    harvest = {json.loads(l)["input_hash"]: json.loads(l)
               for l in args.harvest.read_text(encoding="utf-8").splitlines() if l.strip()}
    teacher = [json.loads(l) for l in args.teacher.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Solo decisioni vere del maestro: niente errori di trasporto o di budget.
    teacher = [t for t in teacher if t["fallback_reason"] in ("", "selected_fallback")
               and t["input_hash"] in harvest]

    repo, sub = DEFAULT_MODELS["english"]
    agent = Agent(repo, device="cuda" if torch.cuda.is_available() else "cpu", subfolder=sub)
    model, tok, device = agent.model, agent.tok, agent.device
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    items = []
    for t in teacher:
        row = harvest[t["input_hash"]]
        q = _question(row)
        seq, markers = build_sequence(tok, row["state"], q, max_len, head_max_len)
        keys = list(q["crit"])
        if len(markers) != len(keys):
            continue
        target = [float(t["probabilities"].get(k, 0.0)) for k in keys]
        s = sum(target)
        if s <= 0:
            continue
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"],
                      "target": [v / s for v in target], "kind": t["kind"],
                      "teacher_choice": keys.index(t["selected"]) if t["selected"] in keys else -1})
    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_fraction))
    val, train = items[:n_val], items[n_val:]
    print(f"[distill] casi: train {len(train)}  val {len(val)}  (device {device})", flush=True)

    # 1) encoder congelato, calcolato una volta
    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    def encode(group):
        b = collate_items([group], tok.pad_token_id)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            h = model.encoder(input_ids=b["input_ids"].to(device),
                              attention_mask=b["attention_mask"].to(device)).last_hidden_state
        return {"h": h.to(torch.float16).cpu(), **{k: b[k] for k in
                ("attention_mask", "marker_pos", "marker_mask", "qtype", "target")}}

    t0 = time.perf_counter()
    batches_train = [encode(train[i:i + args.batch]) for i in range(0, len(train), args.batch)]
    batches_val = [encode(val[i:i + args.batch]) for i in range(0, len(val), args.batch)]
    print(f"[distill] encoder: {time.perf_counter() - t0:.0f} s", flush=True)

    def head_logits(b):
        h = b["h"].to(device).float()
        att = b["attention_mask"].to(device)
        h = h + model.type_emb(b["qtype"].to(device))[:, None, :]
        if model.head is not None:
            pad = ~att.bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        mpos = b["marker_pos"].to(device)
        idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = model.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        return logits.masked_fill(~b["marker_mask"].to(device), -1e4)

    def loss_and_stats(batches, train_mode):
        model.train(train_mode)
        total, n, agree = 0.0, 0, 0
        for b in batches:
            with torch.set_grad_enabled(train_mode):
                logits = head_logits(b)
                logp = F.log_softmax(logits, -1)
                target = b["target"].to(device)
                loss = -(target * logp).sum(-1).mean()
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss) * logits.size(0)
            n += logits.size(0)
            agree += int((logits.argmax(-1) == target.argmax(-1)).sum())
        return total / max(1, n), agree / max(1, n)

    params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    base_val = loss_and_stats(batches_val, False)
    print(f"[distill] prima: val loss {base_val[0]:.4f}  accordo con Jev {base_val[1]:.3f}", flush=True)
    best = (base_val[0], None)
    history = [{"epoch": 0, "val_loss": base_val[0], "val_agreement": base_val[1]}]
    for epoch in range(1, args.epochs + 1):
        tr = loss_and_stats(batches_train, True)
        va = loss_and_stats(batches_val, False)
        history.append({"epoch": epoch, "train_loss": tr[0], "train_agreement": tr[1],
                        "val_loss": va[0], "val_agreement": va[1]})
        print(f"[distill] epoca {epoch}: train {tr[0]:.4f}/{tr[1]:.3f}  val {va[0]:.4f}/{va[1]:.3f}", flush=True)
        if va[0] < best[0]:
            best = (va[0], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                            if not k.startswith("encoder.")})
    if best[1] is not None:
        model.load_state_dict(best[1], strict=False)

    # 2) checkpoint locale: file originali + pesi nuovi + temperature a 1
    src = Path(agent.model_dir) if hasattr(agent, "model_dir") else None
    if src is None or not (src / "rl_agent_config.json").exists():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(repo, allow_patterns=["rl_agent_config.json", "model.safetensors",
                                                           "tokenizer/*", "encoder/*"]))
    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    # Solo il checkpoint inglese: la cache contiene anche gli altri due.
    shutil.copytree(src, out, ignore=shutil.ignore_patterns(
        "model.safetensors", "multilingual", "typed-decisions", ".cache"))
    from safetensors.torch import save_file
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    cfg = json.loads((out / "rl_agent_config.json").read_text(encoding="utf-8"))
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg["temperature_by_options"] = {}
    cfg["distillation"] = {"teacher": "jev (TypeSafe)", "cases_train": len(train), "cases_val": len(val),
                           "epochs": args.epochs, "lr": args.lr, "seed": args.seed, "history": history}
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # 3) valutazione sui 20 casi etichettati a mano, prima e dopo
    report = {"history": history}
    if args.cases.exists():
        data = json.loads(args.cases.read_text(encoding="utf-8"))
        criteria = {
            "sustenance": "Secure food and water: drink, eat, forage, collect ice",
            "resources": "Collect construction materials or minerals",
            "build": "Build or maintain a colony structure",
            "life": "Recover health or rest: medical kit, rest, recovery",
            "explore": "Move, explore or observe new ground",
            "unknown": "Insufficient information to decide",
        }
        distilled = Agent(str(out), device=device.type)
        hits = 0
        rows = []
        for case in data["cases"]:
            res = distilled.system_one(
                {"colony": data["colony"], "colonist": case["colonist"]},
                {"p": {"type": "choice", "instructions": "What should this colonist prioritise this week?",
                       "criteria": criteria}})["answers"]["p"]
            hits += res["choice"] == case["expected"]
            rows.append({"id": case["id"], "expected": case["expected"], "selected": res["choice"],
                         "confidence": res["confidence"]})
        report["hand_cases"] = {"agreement": hits, "cases": len(data["cases"]), "rows": rows}
        print(f"[distill] casi a mano dopo la distillazione: {hits}/{len(data['cases'])}", flush=True)
    (out / "distillation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[distill] checkpoint in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
