#!/usr/bin/env python3
"""Train the learned behavior classifier from examples exported by the browser demo.

    python scripts/train_behavior.py classroom-training-2026-09-09.json [more.json ...] -o data/behavior_model.json

Prints per-class counts and a stratified held-out accuracy, then writes the model JSON that both
the backend (config: default_thresholds.learned.enabled: true) and the demo (Import) can load.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from classroom_monitor.behavior.learned import LABELS, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+")
    ap.add_argument("-o", "--out", default="data/behavior_model.json")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = []
    for f in args.exports:
        d = json.loads(Path(f).read_text())
        examples += [e for e in d["examples"] if e.get("y") in LABELS and isinstance(e.get("x"), list)]
    if not examples:
        raise SystemExit("no examples found")
    counts = Counter(e["y"] for e in examples)
    print("examples:", dict(counts))

    random.seed(args.seed)
    by = {l: [e for e in examples if e["y"] == l] for l in counts}
    train_set, test_set = [], []
    for l, ex in by.items():
        random.shuffle(ex)
        cut = int(len(ex) * args.holdout) if len(ex) >= 5 else 0
        test_set += ex[:cut]
        train_set += ex[cut:]
    labels = [l for l in LABELS if l in counts]
    m = train(np.asarray([e["x"] for e in train_set], float), [e["y"] for e in train_set], labels)
    if test_set:
        ok = sum(m.predict(e["x"])[0] == e["y"] for e in test_set)
        print(f"held-out accuracy: {ok}/{len(test_set)} = {ok / len(test_set):.2%}")
        conf = Counter((e["y"], m.predict(e["x"])[0]) for e in test_set)
        for (y, p), n in sorted(conf.items()):
            print(f"  {y:>12} -> {p:<12} {n}")
    final = train(np.asarray([e["x"] for e in examples], float), [e["y"] for e in examples], labels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": final.to_json(), "n": len(examples), "labels": labels}))
    print("wrote", out)


if __name__ == "__main__":
    main()
