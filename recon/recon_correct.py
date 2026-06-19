#!/usr/bin/env python3
"""
recon_correct.py
================
Standalone module: recover the TRUE seasonal / MAX reading of every metered
channel (circuit breakers + RTU-bearing switches) in a distribution network,
correcting for temporary topology reconfigurations that contaminate the raw
timeseries.

The problem
-----------
A CB or RTU reading is NOT a feeder's intrinsic load. When a neighbouring
outage forces a temporary switching change, load is transferred between
tie-connected feeders: the donor channel reads low, the receiver channel
reads high, for the duration of the reconfiguration. Computing a naive
seasonal MAX over the raw series therefore:
  * INFLATES the receiver's MAX (it briefly carried load that isn't its own)
  * may MISS the donor's true peak (its real load was parked elsewhere)

We never have the historical switch log. All we have is topology + the
channel timeseries. This module reconstructs the native readings using one
physical invariant:

    A reconfiguration is a CONSERVED, ANTI-CORRELATED transfer between
    channels whose feeders are TIE-CONNECTED in the topology.

A genuine load change (a heatwave, a large new customer) shows up on ONE
channel with no compensating partner, so it survives correction untouched.

What is observable
-------------------
Only channels with a physical meter:
  * every circuit breaker  (feeders[*].cb)
  * every switch listed in  feeder_rtus[*][].switch
Plain blocks / DSubs are NOT metered and are never read here; block-level
load remains an estimation problem handled elsewhere (e.g. kVA allocation).

Public API
----------
    topo   = Topology.from_scenario(scenario_dict)
    result = correct_channels(channel_df, topo)         # main entry
    result.native_max          # {channel: corrected MAX amps}
    result.seasonal_max        # {channel: {season: corrected max amps}}
    result.raw_max             # {channel: raw MAX amps}  (for comparison)
    result.events              # DataFrame, one row per detected transfer
    result.native_series       # DataFrame, corrected wide series

CLI
---
    python recon_correct.py <channels.csv> <topology.json> [--report]

channels.csv schema:  channel,timestamp,amps   (long format, hourly)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx


# ===========================================================================
# Topology: which channels exist, and which feeders can exchange load
# ===========================================================================
SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "fall":   (9, 10, 11),
}


@dataclass
class Topology:
    """Everything the corrector needs, derived from a scenario dict."""
    channels: dict          # channel_id -> dict(feeder, kind, downstream)
    cb_of_feeder: dict      # feeder -> CB channel id
    rtus_of_feeder: dict    # feeder -> [rtu channel ids], widest-enclosing first
    tie_adj: dict           # feeder -> set(tie-connected feeders)

    @classmethod
    def from_scenario(cls, sc: dict) -> "Topology":
        edges   = sc["edges"]
        nodes   = sc["nodes"]
        feeders = sc["feeders"]
        rtus    = sc.get("feeder_rtus", {})

        channels = {}
        cb_of_feeder = {}
        for f, fd in feeders.items():
            cb = fd["cb"]
            channels[cb] = dict(feeder=f, kind="cb", downstream=None)
            cb_of_feeder[f] = cb

        rtus_of_feeder = {f: [] for f in feeders}
        for f, entries in rtus.items():
            for e in entries:
                sw = e["switch"]
                channels[sw] = dict(feeder=f, kind="rtu",
                                    downstream=set(e["downstream"]))
                rtus_of_feeder[f].append(sw)
        # widest-enclosing RTU first, so localisation narrows down correctly
        for f in rtus_of_feeder:
            rtus_of_feeder[f].sort(
                key=lambda s: -len(channels[s]["downstream"]))

        # tie adjacency between feeders
        block_feeder = {n: d.get("feeder") for n, d in nodes.items()
                        if d.get("kind") == "block"}
        tie_adj = {f: set() for f in feeders}
        for ed in edges.values():
            if ed["kind"] != "tie":
                continue
            fu = block_feeder.get(ed["u"])
            fv = block_feeder.get(ed["v"])
            if fu and fv and fu != fv:
                tie_adj[fu].add(fv)
                tie_adj[fv].add(fu)

        return cls(channels, cb_of_feeder, rtus_of_feeder, tie_adj)

    @classmethod
    def from_scenario_file(cls, path: str) -> "Topology":
        with open(path) as fh:
            return cls.from_scenario(json.load(fh))

    def cb_channels(self):
        return [c for c, v in self.channels.items() if v["kind"] == "cb"]


# ===========================================================================
# Robust seasonal baseline (so contaminated windows don't bend the fit)
# ===========================================================================
def _design_matrix(ts: pd.Series) -> np.ndarray:
    """Intercept + annual harmonics (3) + daily harmonics (2) + slow drift.

    The slow linear term absorbs a PERMANENT level change (e.g. a large new
    customer connecting) so it is modelled as native load, not as an anomaly.
    """
    t = ts.astype("int64").to_numpy() / 1e9
    doy = ts.dt.dayofyear.to_numpy()
    hod = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    cols = [np.ones(len(ts))]
    for k in (1, 2, 3):
        cols += [np.cos(2 * np.pi * k * doy / 365.25),
                 np.sin(2 * np.pi * k * doy / 365.25)]
    for k in (1, 2):
        cols += [np.cos(2 * np.pi * k * hod / 24.0),
                 np.sin(2 * np.pi * k * hod / 24.0)]
    cols += [t - t.mean()]
    return np.column_stack(cols)


def robust_baseline(ts: pd.Series, y: np.ndarray, iters: int = 6):
    """IRLS Huber fit. Returns (predicted_native, residual, robust_sigma)."""
    X = _design_matrix(ts)
    w = np.ones_like(y)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        WX = X * w[:, None]
        beta, *_ = np.linalg.lstsq(WX, y * w, rcond=None)
        resid = y - X @ beta
        med = np.median(resid)
        sigma = 1.4826 * np.median(np.abs(resid - med)) + 1e-9
        z = (resid - med) / sigma
        w = np.clip(1.0 / np.maximum(np.abs(z), 1.0), 0.05, 1.0)
    pred = X @ beta
    resid = y - pred
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    return pred, resid, sigma


# ===========================================================================
# Result container
# ===========================================================================
@dataclass
class CorrectionResult:
    native_series: pd.DataFrame
    raw_series: pd.DataFrame
    native_max: dict
    raw_max: dict
    seasonal_max: dict
    events: pd.DataFrame
    baseline_pred: pd.DataFrame = field(repr=False, default=None)


# ===========================================================================
# Main entry
# ===========================================================================
def correct_channels(channel_df: pd.DataFrame,
                     topo: Topology,
                     z_thresh: float = 3.0,
                     match_tol: float = 0.35,
                     min_run: int = 6,
                     revert_tol: float = 0.25) -> CorrectionResult:
    """Detect reconfiguration windows and recover native channel readings.

    Parameters
    ----------
    channel_df : long DataFrame [channel, timestamp, amps], hourly.
    topo       : Topology.from_scenario(...).
    z_thresh   : residual sigmas before a CB sample is flagged anomalous.
    match_tol  : how far the neighbour's transfer ratio may stray from the
                 ideal -1 (equal & opposite). 0.35 => accept ratio in
                 roughly [0.65, 1.35] of equal-and-opposite.
    min_run    : minimum consecutive anomalous hours to call it an event
                 (rejects isolated noise spikes).
    revert_tol : a true transfer must REVERT: the feeder's level before and
                 after the window must agree within this fraction. A permanent
                 step (new customer) fails this and is kept as native load.

    Detection runs on CB channels (the feeder-level observable). RTU channels
    are used only to LOCALISE a confirmed transfer within the feeder, never as
    independent evidence (an RTU reading is a subset of its CB reading).
    """
    df = channel_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    wide = df.pivot_table(index="timestamp", columns="channel",
                          values="amps").sort_index()

    # --- robust baseline per channel ---------------------------------------
    pred = pd.DataFrame(index=wide.index)
    res  = pd.DataFrame(index=wide.index)
    sigma = {}
    for ch in wide.columns:
        col = wide[ch]
        mask = col.notna()
        p, r, s = robust_baseline(col[mask].index.to_series(),
                                  col[mask].to_numpy())
        pred[ch] = pd.Series(p, index=col[mask].index)
        res[ch]  = pd.Series(r, index=col[mask].index)
        sigma[ch] = s

    native = wide.copy()
    events = []

    for f in topo.cb_of_feeder:
        cb = topo.cb_of_feeder[f]
        if cb not in res:
            continue
        z = res[cb] / sigma[cb]
        anom = (z.abs() > z_thresh).fillna(False)
        # bridge gaps up to (min_run-1) hours: a single event whose residual
        # momentarily dips below threshold should stay one window, not fragment
        a = anom.to_numpy().copy()
        gap = 0
        for i in range(len(a)):
            if a[i]:
                if 0 < gap < min_run:
                    a[i - gap:i] = True
                gap = 0
            else:
                gap += 1
        anom = pd.Series(a, index=anom.index)
        run_id = (anom != anom.shift()).cumsum()

        for _, block in res[cb].groupby(run_id):
            tr = block.index
            if len(tr) < min_run or not anom.reindex(tr).iloc[0]:
                continue
            r_f = res[cb].reindex(tr)
            mean_dev = float(r_f.mean())

            # magnitude gate: the window's MEAN deviation must be significant
            # at the window level (std error shrinks with sqrt(n)), so a real
            # sustained transfer passes while short noise runs do not.
            def _sig(dev, sg, n):
                return abs(dev) > z_thresh * sg / np.sqrt(n)
            if not _sig(mean_dev, sigma[cb], len(tr)):
                continue

            # --- conservation test vs tie-connected neighbour CBs ----------
            # A sustained transfer shifts each feeder's MEAN oppositely and by
            # a conserved amount. It does NOT make their hourly noise mirror,
            # so we test the window means, not pointwise correlation.
            best_g, best_err, best_ratio = None, np.inf, None
            for g in topo.tie_adj[f]:
                cbg = topo.cb_of_feeder.get(g)
                if cbg is None or cbg not in res:
                    continue
                r_g = res[cbg].reindex(tr)
                if r_g.isna().any():
                    continue
                gdev = float(r_g.mean())
                # partner must move OPPOSITE and by a window-significant amount
                if np.sign(gdev) == np.sign(mean_dev):
                    continue
                if not _sig(gdev, sigma[cbg], len(tr)):
                    continue
                # conserved-magnitude ratio over summed transfer
                ratio = float(-r_g.sum() / r_f.sum()) if r_f.sum() != 0 else np.nan
                if np.isfinite(ratio) and ratio > 0:
                    err = abs(ratio - 1.0)
                    if err < best_err:
                        best_g, best_err, best_ratio = g, err, ratio

            if best_g is None or best_err > match_tol:
                continue  # no conserving partner => genuine load, keep it

            # --- revert test: transfer must return to baseline -------------
            pos = wide.index.get_indexer(tr)
            i0, i1 = pos.min(), pos.max()
            pre = wide[cb].iloc[max(0, i0 - min_run):i0]
            post = wide[cb].iloc[i1 + 1:i1 + 1 + min_run]
            base_level = pred[cb].reindex(tr).mean()
            reverts = True
            for side in (pre, post):
                if len(side) and base_level:
                    if abs(side.mean() - base_level) / abs(base_level) > revert_tol:
                        reverts = False
            if not reverts:
                continue  # permanent step, not a reconfiguration

            # --- confirmed: restore native, localise via RTUs --------------
            native.loc[tr, cb] = pred[cb].reindex(tr).to_numpy()
            located = "upstream-of-all-RTUs (un-localisable below sensors)"
            for rt in topo.rtus_of_feeder.get(f, []):
                if rt in res:
                    rz = (res[rt].reindex(tr) / sigma[rt]).abs().mean()
                    if rz > 2.0:
                        located = f"downstream of {rt}"
                        native.loc[tr, rt] = pred[rt].reindex(tr).to_numpy()
                        break

            events.append(dict(
                feeder=f, channel=cb, partner_feeder=best_g,
                partner_channel=topo.cb_of_feeder[best_g],
                start=tr[0], end=tr[-1], hours=len(tr),
                direction="received" if mean_dev > 0 else "donated",
                mean_amps_dev=round(mean_dev, 1),
                conservation_ratio=round(best_ratio, 3),
                located=located,
            ))

    events_df = pd.DataFrame(events).sort_values("start").reset_index(drop=True) \
        if events else pd.DataFrame(columns=["feeder", "channel", "partner_feeder",
            "partner_channel", "start", "end", "hours", "direction",
            "mean_amps_dev", "conservation_ratio", "located"])

    # --- maxima ------------------------------------------------------------
    raw_max    = {c: float(wide[c].max())    for c in wide.columns}
    native_max = {c: float(native[c].max())  for c in native.columns}

    months = native.index.month
    seasonal = {}
    for c in native.columns:
        seasonal[c] = {}
        for season, ms in SEASON_MONTHS.items():
            sel = native[c][np.isin(months, ms)]
            seasonal[c][season] = float(sel.max()) if len(sel) else float("nan")

    return CorrectionResult(native_series=native, raw_series=wide,
                            native_max=native_max, raw_max=raw_max,
                            seasonal_max=seasonal, events=events_df,
                            baseline_pred=pred)


# ===========================================================================
# CLI
# ===========================================================================
def _report(result: CorrectionResult, topo: Topology):
    print("=== RECONFIGURATION CORRECTION REPORT ===\n")
    print(f"Channels analysed: {len(result.raw_max)}  "
          f"({len(topo.cb_channels())} CB, "
          f"{len(result.raw_max) - len(topo.cb_channels())} RTU)\n")

    if len(result.events):
        print(f"Detected {len(result.events)} reconfiguration window(s):\n")
        for _, e in result.events.iterrows():
            print(f"  {e['channel']:10s} {e['direction']:8s} "
                  f"{abs(e['mean_amps_dev']):6.0f} A  "
                  f"partner {e['partner_channel']:10s}  "
                  f"ratio {e['conservation_ratio']:.2f}  "
                  f"{str(e['start'])[:13]} -> {str(e['end'])[:13]} "
                  f"({e['hours']}h)")
            print(f"             located: {e['located']}")
        print()
    else:
        print("No reconfiguration windows detected.\n")

    print("MAX correction (raw -> native):\n")
    print(f"  {'channel':10s} {'raw':>8s} {'native':>8s} {'delta':>8s}")
    for c in sorted(result.raw_max):
        raw, nat = result.raw_max[c], result.native_max[c]
        flag = "  <-- corrected" if abs(raw - nat) > 1.0 else ""
        print(f"  {c:10s} {raw:8.1f} {nat:8.1f} {nat - raw:+8.1f}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("channels_csv")
    ap.add_argument("topology_json")
    ap.add_argument("--z-thresh", type=float, default=3.0)
    ap.add_argument("--match-tol", type=float, default=0.35)
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=None, help="write corrected wide CSV here")
    args = ap.parse_args()

    topo = Topology.from_scenario_file(args.topology_json)
    df = pd.read_csv(args.channels_csv)
    result = correct_channels(df, topo, z_thresh=args.z_thresh,
                              match_tol=args.match_tol, min_run=args.min_run)
    if args.report:
        _report(result, topo)
    if args.out:
        result.native_series.to_csv(args.out)
        print(f"\nWrote corrected series to {args.out}")


if __name__ == "__main__":
    main()
