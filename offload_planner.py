#!/usr/bin/env python3
"""
Substation Outage Offload Planner (prototype)
==============================================
Given a full substation outage, produce a minimal sequence of OPEN/CLOSE
switch commands that transfers the de-energized feeder sections onto
neighboring energized feeders without violating cable/CB thermal ratings,
while keeping the network radial.

Pipeline:
  1. Synthesize a year of hourly CB load data (stand-in for SCADA history).
  2. Compute seasonal maxima per feeder CB.
  3. Allocate feeder-head load down to switchable load blocks using
     transformer kVA weights, anchored by RTU readings where they exist.
  4. Solve a MILP: maximize restored customers, penalize switching actions
     (manual switches cost more than remote/RTU switches), enforce
     radiality and thermal limits on every edge.
  5. Emit an ordered switching plan (verify CBs open -> dead-zone opens ->
     tie closes) and independently verify the final state.

Dependencies: numpy, networkx, pulp (CBC solver bundled with pulp).
"""

import numpy as np
import networkx as nx
import pulp

RNG = np.random.default_rng(42)

SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "fall":   (9, 10, 11),
}

# ---------------------------------------------------------------------------
# 1. NETWORK MODEL
# ---------------------------------------------------------------------------
# Nodes: substation buses and load blocks (a block = the load between two
# switching devices; granularity is limited to real switch locations).
# Edges: CBs, sectionalizing switches, tie switches, or solid cable runs.
# Only edges with switchable=True can change state. remote=True means the
# device has an RTU / remote control; manual devices cost more to operate.

NODES = {
    # substation buses (sources)
    "SUB-A": dict(kind="bus", headroom_amps=600.0),
    "SUB-B": dict(kind="bus", headroom_amps=600.0),
    "SUB-X": dict(kind="bus", headroom_amps=600.0),   # the outaged substation
    # load blocks: kva = connected transformer kVA (allocation weight)
    "A1-a": dict(kind="block", feeder="A1", kva=1900, customers=820),
    "A1-b": dict(kind="block", feeder="A1", kva=120,  customers=55),
    "B1-a": dict(kind="block", feeder="B1", kva=2300, customers=940),
    "B2-a": dict(kind="block", feeder="B2", kva=1500, customers=610),
    "X1-A": dict(kind="block", feeder="X1", kva=1200, customers=480),
    "X1-B": dict(kind="block", feeder="X1", kva=900,  customers=370),
    "X1-C": dict(kind="block", feeder="X1", kva=800,  customers=320),
    "X2-A": dict(kind="block", feeder="X2", kva=1000, customers=400),
    "X2-B": dict(kind="block", feeder="X2", kva=750,  customers=300),
}

EDGES = {
    # circuit breakers at feeder heads
    "CB-A1":    dict(u="SUB-A", v="A1-a", amp=400, closed=1, switchable=True,  remote=True,  kind="cb"),
    "CB-B1":    dict(u="SUB-B", v="B1-a", amp=400, closed=1, switchable=True,  remote=True,  kind="cb"),
    "CB-B2":    dict(u="SUB-B", v="B2-a", amp=400, closed=1, switchable=True,  remote=True,  kind="cb"),
    "CB-X1":    dict(u="SUB-X", v="X1-A", amp=400, closed=1, switchable=True,  remote=True,  kind="cb"),
    "CB-X2":    dict(u="SUB-X", v="X2-A", amp=400, closed=1, switchable=True,  remote=True,  kind="cb"),
    # solid cable run (no switch), a potential thermal bottleneck on backfeed
    "CABLE-A1": dict(u="A1-a", v="A1-b", amp=200, closed=1, switchable=False, remote=False, kind="cable"),
    # in-feeder sectionalizing switches on the outaged feeders
    "S-X1-AB":  dict(u="X1-A", v="X1-B", amp=250, closed=1, switchable=True,  remote=True,  kind="switch"),
    "S-X1-BC":  dict(u="X1-B", v="X1-C", amp=250, closed=1, switchable=True,  remote=False, kind="switch"),
    "S-X2-AB":  dict(u="X2-A", v="X2-B", amp=250, closed=1, switchable=True,  remote=False, kind="switch"),
    # normally-open tie switches (the backfeed paths)
    "T1":       dict(u="A1-b", v="X1-C", amp=200, closed=0, switchable=True,  remote=True,  kind="tie"),
    "T2":       dict(u="B1-a", v="X1-A", amp=150, closed=0, switchable=True,  remote=False, kind="tie"),
    "T3":       dict(u="B2-a", v="X2-B", amp=200, closed=0, switchable=True,  remote=True,  kind="tie"),
}

FEEDERS = {
    # feeder -> (source CB edge, ordered blocks from CB outward)
    "A1": dict(cb="CB-A1", blocks=["A1-a", "A1-b"]),
    "B1": dict(cb="CB-B1", blocks=["B1-a"]),
    "B2": dict(cb="CB-B2", blocks=["B2-a"]),
    "X1": dict(cb="CB-X1", blocks=["X1-A", "X1-B", "X1-C"]),
    "X2": dict(cb="CB-X2", blocks=["X2-A", "X2-B"]),
}

# Mid-feeder RTUs: switch -> set of blocks DOWNSTREAM of that RTU.
# Sparse on purpose: only X1 has a mid-feeder RTU; everything else is
# allocated purely on transformer kVA.
FEEDER_RTUS = {
    "X1": dict(switch="S-X1-AB", downstream=["X1-B", "X1-C"]),
}

OUTAGE_SUBSTATION = "SUB-X"

# Target summer feeder-head maxima used to scale the synthetic year (amps).
FEEDER_PEAK_TARGET = {"A1": 160.0, "B1": 180.0, "B2": 120.0, "X1": 220.0, "X2": 140.0}

ACTION_COST_REMOTE = 1.0    # SCADA click
ACTION_COST_MANUAL = 5.0    # truck roll
RESTORE_WEIGHT = 100.0      # per customer, dwarfs switching costs


# ---------------------------------------------------------------------------
# 2. SYNTHETIC YEAR OF CB DATA + SEASONAL MAXIMA
# ---------------------------------------------------------------------------
def synthesize_year(feeders, peak_targets):
    """Hourly amps for one non-leap year per feeder CB, summer-peaking."""
    hours = 8760
    month_of_hour = np.repeat(
        np.arange(1, 13),
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    )
    month_of_hour = np.repeat(month_of_hour, 24)
    hod = np.tile(np.arange(24), 365)

    profiles = {}
    for f in feeders:
        daily = 0.65 + 0.35 * np.exp(-((hod - 17.5) ** 2) / 18.0)        # evening peak
        seasonal = 0.75 + 0.25 * np.cos((month_of_hour - 7) * np.pi / 6)  # July peak
        noise = RNG.normal(1.0, 0.04, hours)
        raw = daily * seasonal * noise
        profiles[f] = raw * (peak_targets[f] / raw[month_of_hour == 7].max())
    return profiles, month_of_hour


def seasonal_max(profiles, month_of_hour, season):
    months = SEASON_MONTHS[season]
    mask = np.isin(month_of_hour, months)
    return {f: float(p[mask].max()) for f, p in profiles.items()}


# ---------------------------------------------------------------------------
# 3. LOAD ALLOCATION (CB total -> per-block, RTU-anchored where possible)
# ---------------------------------------------------------------------------
def allocate_block_loads(season_peaks):
    """Split each feeder's seasonal max across its blocks.

    With a mid-feeder RTU we anchor the upstream/downstream split with the
    measured value and distribute within each side by transformer kVA.
    Without one, the whole feeder is distributed by kVA. This is the
    standard pseudo-measurement approach when not every switch is metered.
    """
    block_amps = {}
    for fname, fd in FEEDERS.items():
        total = season_peaks[fname]
        blocks = fd["blocks"]
        kva = {b: NODES[b]["kva"] for b in blocks}

        if fname in FEEDER_RTUS:
            ds = FEEDER_RTUS[fname]["downstream"]
            us = [b for b in blocks if b not in ds]
            # synthetic RTU reading: coincident downstream flow at feeder peak
            ds_true_frac = sum(kva[b] for b in ds) / sum(kva.values())
            rtu_reading = total * ds_true_frac * RNG.normal(1.0, 0.02)
            rtu_reading = min(rtu_reading, total)
            for side, amount in ((us, total - rtu_reading), (ds, rtu_reading)):
                w = sum(kva[b] for b in side)
                for b in side:
                    block_amps[b] = amount * kva[b] / w
        else:
            w = sum(kva.values())
            for b in blocks:
                block_amps[b] = total * kva[b] / w
    return block_amps


# ---------------------------------------------------------------------------
# 4. MILP RESTORATION MODEL
# ---------------------------------------------------------------------------
def solve_offload(block_amps, outage_sub, verbose=False):
    prob = pulp.LpProblem("substation_offload", pulp.LpMaximize)

    nodes = list(NODES)
    edges = list(EDGES)
    buses = [n for n, d in NODES.items() if d["kind"] == "bus"]
    healthy_buses = [b for b in buses if b != outage_sub]
    outage_blocks = [
        n for n, d in NODES.items()
        if d["kind"] == "block" and FEEDERS[d["feeder"]]["cb"] in
        [e for e, ed in EDGES.items() if ed["u"] == outage_sub or ed["v"] == outage_sub]
    ]
    outage_cbs = [e for e, ed in EDGES.items()
                  if ed["kind"] == "cb" and outage_sub in (ed["u"], ed["v"])]

    demand = {n: block_amps.get(n, 0.0) for n in nodes}

    # --- variables -------------------------------------------------------
    x = {e: pulp.LpVariable(f"x_{e}", cat="Binary") for e in edges}        # closed
    y = {n: pulp.LpVariable(f"y_{n}", cat="Binary") for n in nodes}        # energized
    f = {e: pulp.LpVariable(f"f_{e}",                                      # signed amps u->v
                            lowBound=-EDGES[e]["amp"],
                            upBound=EDGES[e]["amp"]) for e in edges}
    g = {b: pulp.LpVariable(f"g_{b}", lowBound=0,                          # bus injection
                            upBound=NODES[b]["headroom_amps"]) for b in buses}
    a = {e: pulp.LpVariable(f"a_{e}", lowBound=0) for e in edges}          # |state change|

    # --- fixed states ------------------------------------------------------
    for e in outage_cbs:
        prob += x[e] == 0                       # de-energized substation: CBs open
    prob += g[outage_sub] == 0
    prob += y[outage_sub] == 0
    for b in healthy_buses:
        prob += y[b] == 1
    for n, d in NODES.items():
        if d["kind"] == "block" and n not in outage_blocks:
            prob += y[n] == 1                   # healthy customers stay on
    for e, ed in EDGES.items():
        if not ed["switchable"]:
            prob += x[e] == ed["closed"]        # solid cable cannot change

    # --- physics & topology ------------------------------------------------
    for n in nodes:
        inflow = pulp.lpSum(f[e] for e, ed in EDGES.items() if ed["v"] == n) \
               - pulp.lpSum(f[e] for e, ed in EDGES.items() if ed["u"] == n)
        inj = g[n] if n in buses else 0
        prob += inflow + inj == demand[n] * y[n], f"kcl_{n}"

    for e, ed in EDGES.items():
        prob += f[e] <= ed["amp"] * x[e]
        prob += f[e] >= -ed["amp"] * x[e]
        prob += x[e] <= y[ed["u"]]              # a closed edge implies live ends
        prob += x[e] <= y[ed["v"]]

    # radiality: closed edges form a spanning forest rooted at healthy buses
    prob += pulp.lpSum(x.values()) == pulp.lpSum(y.values()) - len(healthy_buses)

    # --- switching effort ----------------------------------------------------
    cost = {}
    for e, ed in EDGES.items():
        if e in outage_cbs or not ed["switchable"]:
            cost[e] = 0.0                       # already open / not operable
            prob += a[e] == 0
            continue
        cost[e] = ACTION_COST_REMOTE if ed["remote"] else ACTION_COST_MANUAL
        prob += a[e] >= x[e] - ed["closed"]
        prob += a[e] >= ed["closed"] - x[e]

    prob += (
        RESTORE_WEIGHT * pulp.lpSum(NODES[n]["customers"] * y[n] for n in outage_blocks)
        - pulp.lpSum(cost[e] * a[e] for e in edges)
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=verbose))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"solver status: {pulp.LpStatus[status]}")

    return dict(
        x={e: int(round(x[e].value())) for e in edges},
        y={n: int(round(y[n].value())) for n in nodes},
        flow={e: f[e].value() for e in edges},
        outage_cbs=outage_cbs,
        outage_blocks=outage_blocks,
        healthy_buses=healthy_buses,
    )


# ---------------------------------------------------------------------------
# 5. PLAN EXTRACTION, ORDERING, PAIRING
# ---------------------------------------------------------------------------
def build_plan(sol, block_amps):
    opens, closes = [], []
    for e, ed in EDGES.items():
        if e in sol["outage_cbs"] or not ed["switchable"]:
            continue
        if sol["x"][e] != ed["closed"]:
            (closes if sol["x"][e] == 1 else opens).append(e)

    # which blocks does each closed tie pick up, and from which source?
    G = nx.Graph()
    G.add_nodes_from(n for n in NODES if sol["y"][n])
    for e, ed in EDGES.items():
        if sol["x"][e]:
            G.add_edge(ed["u"], ed["v"], id=e)

    pickup = {}
    for t in closes:
        ed = EDGES[t]
        # the outage-side endpoint of the tie
        out_end = ed["v"] if ed["v"] in sol["outage_blocks"] else ed["u"]
        healthy_end = ed["u"] if out_end == ed["v"] else ed["v"]
        H = G.copy()
        H.remove_edge(ed["u"], ed["v"])
        island = nx.node_connected_component(H, out_end)
        blocks = sorted(b for b in island if b in sol["outage_blocks"])
        src = next(b for b in sol["healthy_buses"]
                   if nx.has_path(G, b, healthy_end))
        feeder = NODES[healthy_end].get("feeder", "?")
        amps = sum(block_amps[b] for b in blocks)
        pickup[t] = dict(blocks=blocks, source=src, via_feeder=feeder, amps=amps)
    return opens, closes, pickup, G


def verify(sol, block_amps, G):
    """Independent checks on the final switched state."""
    report = []

    # 1. every outage CB is open
    ok = all(sol["x"][e] == 0 for e in sol["outage_cbs"])
    report.append(("outage substation CBs open", ok))

    # 2. radial (no loops) and no substation paralleling
    forest = nx.is_forest(G)
    one_src = all(
        sum(1 for n in comp if NODES[n]["kind"] == "bus") <= 1
        for comp in nx.connected_components(G)
    )
    report.append(("final topology is radial (forest)", forest))
    report.append(("no component ties two substations together", one_src))

    # 3. recompute edge currents from the tree and check thermal limits
    loading = {}
    for comp in nx.connected_components(G):
        srcs = [n for n in comp if NODES[n]["kind"] == "bus"]
        if not srcs:
            continue
        T = nx.bfs_tree(G.subgraph(comp), srcs[0])
        for u, v in T.edges():
            eid = G[u][v]["id"]
            downstream = nx.descendants(T, v) | {v}
            amps = sum(block_amps.get(n, 0.0) for n in downstream)
            loading[eid] = (amps, EDGES[eid]["amp"])
    thermal_ok = all(amps <= cap + 1e-6 for amps, cap in loading.values())
    report.append(("all sections within thermal rating", thermal_ok))

    # 4. restoration accounting
    restored = [b for b in sol["outage_blocks"] if sol["y"][b]]
    shed = [b for b in sol["outage_blocks"] if not sol["y"][b]]
    return report, loading, restored, shed


def explain_single_tie_rejections(block_amps):
    """Show why a one-close solution was not enough, where applicable."""
    notes = []
    for t, ed in EDGES.items():
        if ed["kind"] != "tie":
            continue
        out_end = ed["v"] if NODES[ed["v"]].get("feeder", "").startswith("X") else ed["u"]
        feeder = NODES[out_end]["feeder"]
        total = sum(block_amps[b] for b in FEEDERS[feeder]["blocks"])
        if total > ed["amp"]:
            notes.append(
                f"closing {t} alone would push {total:.0f} A through a "
                f"{ed['amp']:.0f} A tie -> a split (extra open) is forced"
            )
    return notes


# ---------------------------------------------------------------------------
# 6. NETWORK DIAGRAM (ADMS-STYLE ONE-LINE VIEW)
# ---------------------------------------------------------------------------
def draw_network(sol=None, block_amps=None):
    """
    Render an ADMS-style one-line diagram using strictly orthogonal routing.

    sol        : result dict from solve_offload(); when None shows the
                 initial outage state (X feeders dead, ties open).
    block_amps : block-name -> amps; annotates each load block when given.

    All connections are horizontal or vertical only.  Where a tie switch
    must cross a feeder segment without connecting, a small semicircle
    flyover (hop) is drawn on the tie to indicate the lines are independent.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        from matplotlib.patches import Arc
    except ImportError:
        raise ImportError(
            "matplotlib is required for draw_network(). pip install matplotlib"
        )

    BG = "#151525"
    HOP_R = 0.12  # radius of the crossing-over semicircle

    # ── Node positions ───────────────────────────────────────────────────────
    # A1-b is at y=1.8 (same as X1-C) so T1 is a single horizontal run.
    # B2-a is at y=3.0 (same as X1-B / X2-B) giving T3 a horizontal channel.
    POS = {
        "SUB-A": (1.0, 5.5), "SUB-B": (5.5, 5.5), "SUB-X":  (9.5, 5.5),
        "A1-a":  (1.0, 4.2), "A1-b":  (1.0, 1.8),
        "B1-a":  (4.5, 4.2), "B2-a":  (6.5, 3.0),
        "X1-A":  (8.5, 4.2), "X1-B":  (8.5, 3.0), "X1-C":  (8.5, 1.8),
        "X2-A": (10.5, 4.2), "X2-B": (10.5, 3.0),
    }

    FEEDER_COLORS = {
        "A1": "#1a78c2", "B1": "#27ae60", "B2": "#16a2b8",
        "X1": "#e07b00", "X2": "#cc3333",
    }

    # ── Explicit orthogonal routes (ordered waypoint lists) ──────────────────
    # CBs fanning out from a substation use an L-route: [start, corner, end].
    # T3 dips to y=2.7 so its horizontal run clears X1-B at (8.5, 3.0).
    ROUTES = {
        "CB-A1":    [(1.0, 5.5), (1.0, 4.2)],
        "CABLE-A1": [(1.0, 4.2), (1.0, 1.8)],
        "CB-B1":    [(5.5, 5.5), (4.5, 5.5), (4.5, 4.2)],
        "CB-B2":    [(5.5, 5.5), (6.5, 5.5), (6.5, 3.0)],
        "CB-X1":    [(9.5, 5.5), (8.5, 5.5), (8.5, 4.2)],
        "CB-X2":    [(9.5, 5.5), (10.5, 5.5), (10.5, 4.2)],
        "S-X1-AB":  [(8.5, 4.2), (8.5, 3.0)],
        "S-X1-BC":  [(8.5, 3.0), (8.5, 1.8)],
        "S-X2-AB":  [(10.5, 4.2), (10.5, 3.0)],
        "T1":       [(1.0, 1.8), (8.5, 1.8)],
        "T2":       [(4.5, 4.2), (8.5, 4.2)],
        "T3":       [(6.5, 3.0), (6.5, 2.7), (10.5, 2.7), (10.5, 3.0)],
    }

    # Crossings where a tie arcs over a feeder segment.
    # The feeder draws straight through; the tie carries the hop.
    #   T2 crosses CB-B2's vertical at (6.5, 4.2)
    #   T3 crosses S-X1-BC's vertical at (8.5, 2.7)
    HOPS = {
        "T2": [(6.5, 4.2)],
        "T3": [(8.5, 2.7)],
    }

    # Explicit switch-symbol positions, chosen to avoid crossings and corners.
    SYMBOL_POS = {
        "CB-A1":   (1.0,  4.85),
        "CB-B1":   (4.5,  4.85),
        "CB-B2":   (6.5,  4.25),
        "CB-X1":   (8.5,  4.85),
        "CB-X2":  (10.5,  4.85),
        "S-X1-AB": (8.5,  3.60),
        "S-X1-BC": (8.5,  2.40),
        "S-X2-AB": (10.5, 3.60),
        "T1":      (4.75, 1.8),
        "T2":      (5.5,  4.2),   # left of the CB-B2 crossing
        "T3":      (7.5,  2.7),   # between the bend and the S-X1-BC crossing
    }

    # ── Switch / energization states ─────────────────────────────────────────
    if sol is not None:
        xstate = sol["x"]
        ystate = sol["y"]
    else:
        xstate = {e: ed["closed"] for e, ed in EDGES.items()}
        outage_cbs = {e for e, ed in EDGES.items() if ed["u"] == OUTAGE_SUBSTATION}
        outage_feeders = {
            NODES[ed["v"]]["feeder"] for e, ed in EDGES.items()
            if e in outage_cbs and NODES[ed["v"]]["kind"] == "block"
        }
        ystate = {
            n: 0 if (n == OUTAGE_SUBSTATION or
                     (NODES[n]["kind"] == "block" and
                      NODES[n]["feeder"] in outage_feeders))
            else 1
            for n in NODES
        }

    # ── Canvas ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0.5, 7.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    def _draw_route(pts, hops_xy, color, lw, ls, zorder):
        """
        Draw an orthogonal polyline, inserting a semicircle flyover at each
        hop point.  Hops arch upward on horizontal segments.
        """
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            is_h = abs(y1 - y0) < 1e-9

            # Collect hop positions that fall on this horizontal segment.
            seg_hops = []
            if is_h:
                for hx, hy in hops_xy:
                    if (abs(hy - y0) < 1e-9 and
                            min(x0, x1) + 1e-9 < hx < max(x0, x1) - 1e-9):
                        seg_hops.append(hx)

            if not seg_hops:
                ax.plot([x0, x1], [y0, y1], color=color, lw=lw, linestyle=ls,
                        solid_capstyle="butt", zorder=zorder)
            else:
                seg_hops.sort(reverse=(x0 > x1))
                cur = x0
                for hx in seg_hops:
                    ax.plot([cur, hx - HOP_R], [y0, y0],
                            color=color, lw=lw, linestyle=ls,
                            solid_capstyle="butt", zorder=zorder)
                    # Semicircle arching upward over the crossing feeder.
                    ax.add_patch(Arc(
                        (hx, y0), 2 * HOP_R, 2 * HOP_R,
                        angle=0, theta1=0, theta2=180,
                        color=color, lw=lw, zorder=zorder + 1,
                    ))
                    cur = hx + HOP_R
                ax.plot([cur, x1], [y0, y1], color=color, lw=lw, linestyle=ls,
                        solid_capstyle="butt", zorder=zorder)

    # ── Draw edges ────────────────────────────────────────────────────────────
    for eid, ed in EDGES.items():
        is_closed = bool(xstate[eid])
        feeder = NODES[ed["u"]].get("feeder") or NODES[ed["v"]].get("feeder")
        line_color = FEEDER_COLORS.get(feeder, "#aaaaaa") if is_closed else "#445566"
        lw = 3.0 if ed["kind"] in ("cb", "cable") else 2.0
        ls = "--" if ed["kind"] == "tie" else "-"

        _draw_route(ROUTES[eid], HOPS.get(eid, []),
                    line_color, lw, ls, zorder=2)

        # Switch symbol: filled circle = closed, circled-X = open.
        if ed["switchable"]:
            sx, sy = SYMBOL_POS[eid]
            sym_color = "#dddddd" if is_closed else "#ff6b6b"
            r = 0.13
            ax.add_patch(plt.Circle(
                (sx, sy), r, facecolor=BG, edgecolor=sym_color, lw=2.5, zorder=5
            ))
            if is_closed:
                ax.add_patch(plt.Circle((sx, sy), r * 0.4, color=sym_color, zorder=6))
            else:
                r2 = r * 0.55
                ax.plot([sx - r2, sx + r2], [sy - r2, sy + r2],
                        color=sym_color, lw=1.8, zorder=6)
                ax.plot([sx - r2, sx + r2], [sy + r2, sy - r2],
                        color=sym_color, lw=1.8, zorder=6)

            ax.text(sx + 0.18, sy + 0.18, eid,
                    fontsize=6.5, ha="left", va="bottom", color="#bbbbcc",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=BG,
                              edgecolor="none", alpha=0.65),
                    zorder=7)
        else:
            # Label at midpoint of route for non-switchable edges (CABLE-A1).
            pts = ROUTES[eid]
            mx = (pts[0][0] + pts[-1][0]) / 2
            my = (pts[0][1] + pts[-1][1]) / 2
            ax.text(mx + 0.18, my, eid,
                    fontsize=6.5, ha="left", va="center", color="#888899",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=BG,
                              edgecolor="none", alpha=0.65),
                    zorder=7)

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for nid, nd in NODES.items():
        x, y = POS[nid]
        energized = bool(ystate.get(nid, 1))

        if nd["kind"] == "bus":
            fc = "#c0392b" if nid == OUTAGE_SUBSTATION else "#2471a3"
            if not energized:
                fc = "#3a3a4a"
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - 0.52, y - 0.21), 1.04, 0.42,
                boxstyle="round,pad=0.06",
                facecolor=fc, edgecolor="#888899", lw=1.5, zorder=8,
            ))
            tag = " [OFF]" if not energized else ""
            ax.text(x, y, nid + tag, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=9)
        else:
            fc = FEEDER_COLORS.get(nd["feeder"], "#666") if energized else "#2e2e3e"
            ax.add_patch(plt.Circle(
                (x, y), 0.34, facecolor=fc, edgecolor="#888899", lw=1.5, zorder=8
            ))
            ax.text(x, y + 0.08, nid, ha="center", va="center",
                    fontsize=6.5, fontweight="bold",
                    color="white" if energized else "#666677", zorder=9)
            ax.text(x, y - 0.12, f"{nd['customers']}c",
                    ha="center", va="center", fontsize=6,
                    color="white" if energized else "#666677", zorder=9)
            if block_amps and nid in block_amps:
                ax.text(x + 0.42, y + 0.25, f"{block_amps[nid]:.0f} A",
                        fontsize=6.5, color="#f0c040",
                        ha="left", va="bottom", zorder=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    leg_items = [
        mpatches.Patch(facecolor="#2471a3", edgecolor="#888899",
                       label="Healthy substation bus"),
        mpatches.Patch(facecolor="#c0392b", edgecolor="#888899",
                       label="Outaged substation (SUB-X)"),
        *[mpatches.Patch(facecolor=c, label=f"Feeder {f}")
          for f, c in FEEDER_COLORS.items()],
        mlines.Line2D([0], [0], color="#888899", lw=2, ls="--",
                      label="Tie switch (dashed)"),
        mlines.Line2D([0], [0], color="w", marker="o", markerfacecolor=BG,
                      markeredgecolor="#dddddd", markersize=12, markeredgewidth=2.5,
                      label="Switch closed (●)"),
        mlines.Line2D([0], [0], color="w", marker="o", markerfacecolor=BG,
                      markeredgecolor="#ff6b6b", markersize=12, markeredgewidth=2.5,
                      label="Switch open (×)"),
        mlines.Line2D([0], [0], color="#888899", lw=2,
                      label="⌢  crossing — not a junction"),
    ]
    leg = ax.legend(handles=leg_items, loc="lower right", fontsize=8,
                    facecolor="#1e1e30", edgecolor="#444455", labelcolor="white",
                    title="Legend", title_fontsize=8)
    leg.get_title().set_color("#ccccdd")

    mode = "Post-switching state" if sol else "Initial outage state"
    ax.set_title(
        f"Substation Outage Offload Planner  ·  {mode}",
        fontsize=13, fontweight="bold", color="white", pad=14,
    )

    plt.tight_layout()
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(season="summer"):
    profiles, moh = synthesize_year(FEEDERS, FEEDER_PEAK_TARGET)
    peaks = seasonal_max(profiles, moh, season)
    block_amps = allocate_block_loads(peaks)

    print(f"=== SUBSTATION OUTAGE OFFLOAD PLAN  (season: {season}) ===")
    print(f"Outage: {OUTAGE_SUBSTATION} fully de-energized\n")

    print("Seasonal feeder maxima (from 1 yr of CB data):")
    for fn, v in peaks.items():
        print(f"  {fn}: {v:6.1f} A")
    print("\nAllocated block loads (kVA-weighted, RTU-anchored where present):")
    for b, v in block_amps.items():
        tag = ""
        fd = NODES[b]["feeder"]
        if fd in FEEDER_RTUS and b in FEEDER_RTUS[fd]["downstream"]:
            tag = "  [below RTU " + FEEDER_RTUS[fd]["switch"] + "]"
        print(f"  {b}: {v:6.1f} A{tag}")

    sol = solve_offload(block_amps, OUTAGE_SUBSTATION)
    opens, closes, pickup, G = build_plan(sol, block_amps)

    for note in explain_single_tie_rejections(block_amps):
        print(f"\nNote: {note}")

    print("\n--- SWITCHING SEQUENCE ---")
    step = 0
    for e in sol["outage_cbs"]:
        step += 1
        print(f"Step {step}: VERIFY OPEN  {e}  (feeder breaker, substation de-energized)")
    for e in opens:
        step += 1
        ed = EDGES[e]
        mode = "remote" if ed["remote"] else "MANUAL"
        print(f"Step {step}: OPEN   {e}  ({mode})  sectionalize dead zone "
              f"between {ed['u']} | {ed['v']}")
    for e in closes:
        step += 1
        ed = EDGES[e]
        mode = "remote" if ed["remote"] else "MANUAL"
        p = pickup[e]
        print(f"Step {step}: CLOSE  {e}  ({mode})  backfeed {', '.join(p['blocks'])} "
              f"from {p['source']} via feeder {p['via_feeder']}  "
              f"(~{p['amps']:.0f} A pickup)")

    print("\n--- OPEN/CLOSE PAIRING (radial integrity) ---")
    for e in closes:
        p = pickup[e]
        boundary = [o for o in opens + sol["outage_cbs"]
                    if EDGES[o]["u"] in p["blocks"] or EDGES[o]["v"] in p["blocks"]]
        print(f"  CLOSE {e} is bounded by OPEN {', '.join(boundary)}")

    report, loading, restored, shed = verify(sol, block_amps, G)
    print("\n--- VERIFICATION ---")
    for name, ok in report:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    cust_total = sum(NODES[b]["customers"] for b in sol["outage_blocks"])
    cust_rest = sum(NODES[b]["customers"] for b in restored)
    print(f"\nRestored {len(restored)}/{len(sol['outage_blocks'])} blocks, "
          f"{cust_rest}/{cust_total} customers ({100*cust_rest/cust_total:.0f}%)")
    if shed:
        print(f"Unrestorable (shed): {', '.join(shed)}")

    print("\nPost-switching section loading:")
    for e, (amps, cap) in sorted(loading.items(), key=lambda kv: -kv[1][0]/kv[1][1]):
        print(f"  {e:9s} {amps:6.1f} / {cap:.0f} A  ({100*amps/cap:5.1f}%)")

    n_remote = sum(1 for e in opens + closes if EDGES[e]["remote"])
    n_manual = len(opens) + len(closes) - n_remote
    print(f"\nSwitching effort: {len(opens)} open + {len(closes)} close "
          f"({n_remote} remote, {n_manual} manual)")

    try:
        draw_network(sol=sol, block_amps=block_amps)
    except ImportError:
        print("\n(Install matplotlib to view the network diagram: pip install matplotlib)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "summer")