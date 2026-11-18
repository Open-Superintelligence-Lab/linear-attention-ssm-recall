"""Markdown tables for the official-code baselines: MQAR (zoology) and RULER-style NIAH.

python3 benchmark/official_table.py            # reads data/runs/mqar/run_*.json, data/runs/ruler/run_*.json
python3 benchmark/official_table.py --fill     # writes both tables into BASELINES.md between the markers
"""
import re
import sys
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["attention", "based", "gated_deltanet", "gla", "rwkv7", "mamba2"]


def load(pattern):
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROOT, pattern))):
        try:
            o = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        m = o["config"]["method"]
        if m not in rows or o["score"] > rows[m]["score"]:   # best over the lr settings run
            rows[m] = o
    return [rows[m] for m in ORDER if m in rows]


def table(rows, cols):
    if not rows:
        return "(no runs)\n"
    out = ["| Method | lr | Params | " + " | ".join(cols) + " | Score | Train s | Peak GB |",
           "|---|---|---|" + "---|" * len(cols) + "---|---|---|"]
    for o in rows:
        cells = [f'{o["acc"].get(c, float("nan")):.3f}' for c in cols]
        out.append(f'| {o["config"]["method"]} | {o["config"]["lr"]:g} | {o["params"]:,} | ' + " | ".join(cells)
                   + f' | **{o["score"]:.3f}** | {o["train_secs"]:.0f} | {o["peak_rss_gb"]} |')
    return "\n".join(out) + "\n"


def tables():
    mq = load("data/runs/mqar/run_*.json")
    t_mq = table(mq, list(mq[0]["acc"]) if mq else [])
    ru = load("data/runs/ruler/run_*.json")
    t_ru = "(no runs)\n"
    if ru:
        lens = list(ru[0]["by_len"])
        for o in ru:
            for L in lens:
                o["acc"][f"all@{L}"] = o["by_len"][L]
        tasks = sorted({k.split("@")[0] for k in ru[0]["acc"] if not k.startswith("all")})
        cols = [f"{t}@{lens[0]}" for t in tasks] + [f"all@{L}" for L in lens]
        t_ru = table(ru, cols).replace("niah_", "").replace(f"@{lens[0]} |", " |").replace("| all |", f"| all@{lens[0]} |")
    return t_mq, t_ru


def main():
    t_mq, t_ru = tables()
    if "--fill" in sys.argv:
        p = os.path.join(ROOT, "BASELINES.md")
        s = open(p).read()
        for tag, t in (("mqar", t_mq), ("ruler", t_ru)):
            s = re.sub(f"<!-- {tag} -->.*?<!-- /{tag} -->", f"<!-- {tag} -->\n{t}<!-- /{tag} -->", s, flags=re.S)
        open(p, "w").write(s)
    print(t_mq)
    print(t_ru)


if __name__ == "__main__":
    main()
