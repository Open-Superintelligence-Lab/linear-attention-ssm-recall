"""Read JSON lines from benchmark.run on stdin, print the baseline table as Markdown."""
import json
import sys

rows = [json.loads(l) for l in sys.stdin if l.strip().startswith("{")]
print("| Method | State floats (L=1024) | Params | L128 | L256 | L512 | L1024 | MQAR-LR score | Secs |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    m = r["config"]["method"] + (f" ({r['config']['hybrid']})" if r["config"].get("hybrid") else "")
    if r.get("diverged"):
        print(f"| {m} | {r['state_floats_L1024']} | {r['params']} | diverged at step {r['step']} | | | | | {r['secs']} |")
        continue
    accs = " | ".join(f"{r[f'bench_L{L}']:.3f}" for L in (128, 256, 512, 1024))
    print(f"| {m} | {r['state_floats_L1024']} | {r['params']} | {accs} | {r['bench_score']:.3f} | {r['secs']} |")
