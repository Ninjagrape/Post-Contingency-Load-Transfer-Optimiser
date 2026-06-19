#!/usr/bin/env python3
"""
generate_channel_data.py
========================
Produce 2 years of hourly simulated readings for every metered channel
(CB + RTU) of a scenario, with KNOWN reconfiguration events injected so the
corrector can be scored against ground truth.

Each channel gets a native profile:
    native(t) = peak * daily_shape(hour) * seasonal_shape(month)
                * weekly_dip(weekend) * noise
optionally plus a permanent step partway through (a "new large customer"),
which is a DECOY: it must survive correction as real load.

Reconfiguration events transfer load between TIE-CONNECTED feeders:
    donor CB  -= delta(t)        receiver CB += delta(t)
for a contiguous window. RTU channels beneath the affected segment move too,
so localisation can be tested.

Outputs (into ./data):
    channels_long.csv     channel,timestamp,amps     (feed this to the module)
    channels_wide.csv     timestamp + one column per channel (human view)
    ground_truth.json     the injected events + true native MAX per channel
"""

import json
import os
import numpy as np
import pandas as pd

from recon_correct import Topology

RNG = np.random.default_rng(7)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")
SCENARIOS = os.path.join(ROOT, "scenarios")

# channel peak amps (native, before any transfer). Keyed by channel id.
# CBs carry more than RTUs (which see only a downstream subset).
PEAKS = {
    # metro.json channels
    "CB-X1": 430, "CB-X23": 470, "CB-B1": 360, "CB-C1": 250, "CB-D1": 220,
    "CB-A12": 300,
    "S-X1-AD": 250, "S-X1-FG": 130,
    "S-X2-AD": 240, "S-X2-DE": 150,
}

# Permanent step decoys: channel -> (fraction_into_series, +amps)
# A genuine new large customer. Must NOT be corrected away.
PERMANENT_STEPS = {
    "CB-C1": (0.55, 55.0),
}

# Reconfiguration events: each transfers load donor->receiver for a window.
# (donor_cb, receiver_cb, start 'YYYY-MM-DD', length_hours, amps,
#  rtu_on_donor_that_moves or None)
EVENTS = [
    ("CB-X1", "CB-B1", "2023-02-08", 72, 110.0, "S-X1-FG"),   # winter, localisable
    ("CB-B1", "CB-X1", "2023-07-19", 120, 95.0, None),        # summer, reverse dir
    ("CB-X23", "CB-C1", "2024-03-02", 48, 80.0, "S-X2-DE"),   # spring, year 2
    ("CB-D1", "CB-X23", "2024-09-15", 60, 60.0, None),        # fall, D1 ties X3 on CB-X23
]


def daily_shape(hour):
    # evening-peaking, trough overnight
    return 0.55 + 0.45 * np.exp(-((hour - 18.0) ** 2) / 16.0)


def seasonal_shape(month, kind="summer"):
    # summer-peaking feeders; flip phase for a couple to add variety
    phase = 7 if kind == "summer" else 1
    return 0.70 + 0.30 * np.cos((month - phase) * np.pi / 6.0)


def build():
    os.makedirs(OUT, exist_ok=True)
    # load a topology purely to know which channels exist / sanity-check
    with open(os.path.join(SCENARIOS, "metro.json")) as fh:
        sc = json.load(fh)
    topo = Topology.from_scenario(sc)
    known = set(topo.channels)
    for c in PEAKS:
        assert c in known, f"{c} is not a real channel in metro.json"

    idx = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="h")
    hour = idx.hour.to_numpy()
    month = idx.month.to_numpy()
    dow = idx.dayofweek.to_numpy()
    n = len(idx)

    # winter-peaking set for a little realism
    winter_feeders = {"CB-D1"}

    native = {}
    for ch, peak in PEAKS.items():
        kind = "winter" if ch in winter_feeders else "summer"
        shape = daily_shape(hour) * seasonal_shape(month, kind)
        weekend = np.where(dow >= 5, 0.90, 1.0)
        noise = RNG.normal(1.0, 0.035, n)
        series = peak * shape * weekend * noise
        # permanent step decoy
        if ch in PERMANENT_STEPS:
            frac, amp = PERMANENT_STEPS[ch]
            series[int(frac * n):] += amp
        native[ch] = series

    # keep a pristine copy for ground-truth MAX (native, no transfers)
    native_max_true = {ch: float(np.max(v)) for ch, v in native.items()}

    # inject reconfiguration transfers on top of native
    observed = {ch: v.copy() for ch, v in native.items()}
    gt_events = []
    for donor, recv, start, length, amps, rtu in EVENTS:
        s = idx.get_indexer([pd.Timestamp(start)])[0]
        e = s + length
        # smooth ramp in/out so edges aren't perfectly square
        win = np.ones(length)
        ramp = max(1, length // 8)
        win[:ramp] = np.linspace(0, 1, ramp)
        win[-ramp:] = np.linspace(1, 0, ramp)
        delta = amps * win
        observed[donor][s:e] -= delta
        observed[recv][s:e] += delta
        if rtu and rtu in observed:
            observed[rtu][s:e] -= delta * 0.8   # most of the moved load was below it
        gt_events.append(dict(donor=donor, receiver=recv, start=str(idx[s]),
                              end=str(idx[e - 1]), hours=length, amps=amps,
                              rtu_moved=rtu))

    # assemble long + wide frames
    wide = pd.DataFrame({"timestamp": idx})
    for ch in PEAKS:
        wide[ch] = np.round(observed[ch], 2)
    long = wide.melt(id_vars="timestamp", var_name="channel", value_name="amps")
    long = long.sort_values(["channel", "timestamp"]).reset_index(drop=True)

    wide.to_csv(os.path.join(OUT, "channels_wide.csv"), index=False)
    long.to_csv(os.path.join(OUT, "channels_long.csv"), index=False)
    with open(os.path.join(OUT, "ground_truth.json"), "w") as fh:
        json.dump(dict(events=gt_events,
                       native_max_true=native_max_true,
                       permanent_steps=PERMANENT_STEPS), fh, indent=2)

    print(f"Wrote {len(long)} rows across {len(PEAKS)} channels to {OUT}/")
    print(f"Injected {len(EVENTS)} reconfiguration events; "
          f"{len(PERMANENT_STEPS)} permanent-step decoy(s).")


if __name__ == "__main__":
    build()
