"""One run: train one model from scratch, score it on MQAR-LR, print one JSON line.

python3 -m benchmark.run method=gated_deltanet
python3 -m benchmark.run method=mamba2 head_k=32 conv=0 steps=3000
Any key in env/config.py DEFAULTS can be set as key=value.
"""
import json
import sys
import time

from env.config import resolve
from env.train import train
from benchmark.bench import evaluate


def run(overrides):
    cfg = resolve(overrides)
    t0 = time.time()
    model, info = train(cfg)
    out = dict(config={k: v for k, v in sorted(cfg.items())})
    out["params"] = sum(p.numel() for p in model.parameters())
    out["state_floats_L1024"] = model.state_floats(1024)
    out["layer_types"] = model.layer_types
    out.update({k: v for k, v in info.items() if k != "curve"})
    if info.get("curve"):
        out["curve"] = info["curve"]
    if not info["diverged"]:
        out.update(evaluate(model, cfg))
    out["secs"] = round(time.time() - t0, 1)
    return out


def main(argv):
    overrides = dict(a.split("=", 1) for a in argv if "=" in a)
    print(json.dumps(run(overrides)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
