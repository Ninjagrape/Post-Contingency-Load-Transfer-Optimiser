#!/usr/bin/env python3
"""Score detector output against injected ground truth."""
import json
import os
import pandas as pd
from recon_correct import Topology, correct_channels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

topo = Topology.from_scenario_file(os.path.join(ROOT, "scenarios", "metro.json"))
df = pd.read_csv(os.path.join(DATA, "channels_long.csv"))
gt = json.load(open(os.path.join(DATA, "ground_truth.json")))

res = correct_channels(df, topo)
ev = res.events

print(f"Injected events: {len(gt['events'])}   Detected windows: {len(ev)}\n")

print("INJECTED:")
for e in gt["events"]:
    print(f"  {e['donor']:8s} -> {e['receiver']:8s}  {e['amps']:.0f}A  "
          f"{e['start'][:13]} .. {e['end'][:13]} ({e['hours']}h)")

print("\nDETECTED:")
if len(ev):
    for _, e in ev.iterrows():
        print(f"  {e['channel']:8s} {e['direction']:8s} {abs(e['mean_amps_dev']):.0f}A  "
              f"partner {e['partner_channel']:8s} ratio {e['conservation_ratio']:.2f}  "
              f"{str(e['start'])[:13]} .. {str(e['end'])[:13]} ({e['hours']}h)")

# match detected windows to injected by time overlap
def overlap(a0, a1, b0, b1):
    return max(pd.Timestamp(a0), pd.Timestamp(b0)) <= min(pd.Timestamp(a1), pd.Timestamp(b1))

matched = 0
for e in gt["events"]:
    hit = any(overlap(e["start"], e["end"], r["start"], r["end"]) for _, r in ev.iterrows()) if len(ev) else False
    matched += hit
print(f"\nInjected events with a detection overlap: {matched}/{len(gt['events'])}")

print("\nMAX recovery (native_true vs corrected vs raw):")
print(f"  {'channel':10s} {'true':>8s} {'corrected':>10s} {'raw':>8s}")
for c in sorted(res.native_max):
    t = gt["native_max_true"].get(c, float('nan'))
    print(f"  {c:10s} {t:8.1f} {res.native_max[c]:10.1f} {res.raw_max[c]:8.1f}")
