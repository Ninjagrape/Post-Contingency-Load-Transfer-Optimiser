#!/usr/bin/env python3
"""
Substation Outage Offload Planner v2
=====================================
Merges v1 (ADMS-style one-line diagram, switching plan, single-tie rejection
analysis) with v2's advanced features:

  * Substations with MULTIPLE circuit breakers / feeders.
  * Feeders that BRANCH (tee-offs / laterals) rather than running as a stub.
  * Intra-feeder RING configurations with relocatable normally-open point
    (penalized in the objective to discourage unnecessary ring moves).
  * MULTIPLE inter-substation tie points per feeder.
  * Connectivity-flow radiality (necessary AND sufficient), replacing the v1
    edge-count constraint which a meshed network can silently defeat.
  * Tree-aware load allocation: pushes each CB reading down the feeder tree,
    subtracts RTU-measured branch flows, spreads the residual by kVA over
    unmeasured blocks only.

Pipeline:
  1. Synthesize a year of hourly CB data (stand-in for SCADA history).
  2. Compute seasonal maxima per feeder CB.
  3. Allocate feeder-head load to blocks (kVA-weighted, RTU-anchored).
  4. Solve MILP: maximise restored customers, penalise switching actions
     (manual > remote, ring relocation adds extra penalty), enforce
     connectivity-flow radiality and thermal limits.
  5. Emit an ordered switching plan, flag over-loaded ties, and verify
     the final state independently.
  6. Render an ADMS-style one-line diagram (requires matplotlib).

Dependencies: numpy, networkx, pulp (bundled CBC), matplotlib (optional).
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

# ===========================================================================
# NETWORK MODEL
# ===========================================================================
# Three substations:
#   SUB-X : OUTAGED. Two CBs -> feeder X1 (ring + tee) and X2 (tee).
#   SUB-A : healthy.  Feeders A1 (with lateral) and A2.
#   SUB-B : healthy.  Feeder B1 (with lateral).
#
# X1 is a RING: leaves CB-X1, branches at X1-A into leg-1 (X1-B -> X1-C)
# and leg-2 tee (X1-D -> X1-E -> X1-F). The ring is closed through R-X1
# between X1-C and X1-F (normally open today -> radial; planner may relocate
# the open point at an extra penalty cost).
#
# Multiple tie points (all normally open):
#   T1: A1-b  <-> X1-C   (SUB-A backfeeds X1 leg-1 tail)
#   T2: B1-a  <-> X1-E   (SUB-B backfeeds X1 leg-2)
#   T3: A2-a  <-> X2-C   (SUB-A backfeeds X2 lateral)
#   T4: B1-b  <-> X2-B   (SUB-B backfeeds X2 main)

NODES = {
    # substation buses (sources)
    "SUB-X": dict(kind="bus", headroom_amps=900.0),
    "SUB-A": dict(kind="bus", headroom_amps=900.0),
    "SUB-B": dict(kind="bus", headroom_amps=900.0),
    # X1 feeder (ring + tee): kva = connected transformer kVA (allocation weight)
    "X1-A": dict(kind="block", feeder="X1", kva=900,  customers=360),
    "X1-B": dict(kind="block", feeder="X1", kva=700,  customers=280),  # leg-1
    "X1-C": dict(kind="block", feeder="X1", kva=600,  customers=240),  # leg-1 tail
    "X1-D": dict(kind="block", feeder="X1", kva=650,  customers=260),  # leg-2 tee
    "X1-E": dict(kind="block", feeder="X1", kva=500,  customers=200),  # leg-2
    "X1-F": dict(kind="block", feeder="X1", kva=400,  customers=160),  # ring join
    # X2 feeder (tee)
    "X2-A": dict(kind="block", feeder="X2", kva=800,  customers=320),
    "X2-B": dict(kind="block", feeder="X2", kva=600,  customers=240),  # main
    "X2-C": dict(kind="block", feeder="X2", kva=450,  customers=180),  # lateral
    # SUB-A feeders
    "A1-a": dict(kind="block", feeder="A1", kva=1100, customers=440),
    "A1-b": dict(kind="block", feeder="A1", kva=300,  customers=120),  # lateral
    "A2-a": dict(kind="block", feeder="A2", kva=950,  customers=380),
    # SUB-B feeder
    "B1-a": dict(kind="block", feeder="B1", kva=1200, customers=480),
    "B1-b": dict(kind="block", feeder="B1", kva=350,  customers=140),  # lateral
}

EDGES = {
    # feeder-head circuit breakers
    "CB-X1":   dict(u="SUB-X", v="X1-A", amp=420, closed=1, switchable=True,  remote=True,  kind="cb"),
    "CB-X2":   dict(u="SUB-X", v="X2-A", amp=420, closed=1, switchable=True,  remote=True,  kind="cb"),
    "CB-A1":   dict(u="SUB-A", v="A1-a", amp=420, closed=1, switchable=True,  remote=True,  kind="cb"),
    "CB-A2":   dict(u="SUB-A", v="A2-a", amp=420, closed=1, switchable=True,  remote=True,  kind="cb"),
    "CB-B1":   dict(u="SUB-B", v="B1-a", amp=480, closed=1, switchable=True,  remote=True,  kind="cb"),
    # X1 ring + tee internals
    "S-X1-AB": dict(u="X1-A", v="X1-B", amp=300, closed=1, switchable=True,  remote=True,  kind="switch"),
    "S-X1-BC": dict(u="X1-B", v="X1-C", amp=260, closed=1, switchable=True,  remote=False, kind="switch"),
    "S-X1-AD": dict(u="X1-A", v="X1-D", amp=300, closed=1, switchable=True,  remote=False, kind="switch"),
    "S-X1-DE": dict(u="X1-D", v="X1-E", amp=260, closed=1, switchable=True,  remote=True,  kind="switch"),
    "S-X1-EF": dict(u="X1-E", v="X1-F", amp=220, closed=1, switchable=True,  remote=False, kind="switch"),
    "R-X1":    dict(u="X1-C", v="X1-F", amp=220, closed=0, switchable=True,  remote=True,  kind="ring"),
    # X2 tee internals
    "S-X2-AB": dict(u="X2-A", v="X2-B", amp=300, closed=1, switchable=True,  remote=True,  kind="switch"),
    "S-X2-BC": dict(u="X2-B", v="X2-C", amp=240, closed=1, switchable=False, remote=False, kind="cable"),
    # healthy feeder internals (solid cable, cannot switch)
    "S-A1-ab": dict(u="A1-a", v="A1-b", amp=240, closed=1, switchable=False, remote=False, kind="cable"),
    "S-B1-ab": dict(u="B1-a", v="B1-b", amp=260, closed=1, switchable=False, remote=False, kind="cable"),
    # normally-open inter-substation ties
    "T1":      dict(u="A1-b", v="X1-C", amp=220, closed=0, switchable=True,  remote=True,  kind="tie"),
    "T2":      dict(u="B1-a", v="X1-E", amp=200, closed=0, switchable=True,  remote=False, kind="tie"),
    "T3":      dict(u="A2-a", v="X2-C", amp=240, closed=0, switchable=True,  remote=True,  kind="tie"),
    "T4":      dict(u="B1-b", v="X2-B", amp=200, closed=0, switchable=True,  remote=True,  kind="tie"),
}

FEEDERS = {
    "X1": dict(cb="CB-X1", root="X1-A"),
    "X2": dict(cb="CB-X2", root="X2-A"),
    "A1": dict(cb="CB-A1", root="A1-a"),
    "A2": dict(cb="CB-A2", root="A2-a"),
    "B1": dict(cb="CB-B1", root="B1-a"),
}

# Mid-feeder RTUs: list of {switch, downstream} entries per feeder.
# downstream = blocks on the far side of that RTU on the normal radial tree.
FEEDER_RTUS = {
    "X1": [dict(switch="S-X1-AD", downstream=["X1-D", "X1-E", "X1-F"])],
    "X2": [dict(switch="S-X2-AB", downstream=["X2-B", "X2-C"])],
}

OUTAGE_SUBSTATION = "SUB-X"

FEEDER_PEAK_TARGET = {
    "X1": 360.0, "X2": 250.0,
    "A1": 230.0, "A2": 150.0, "B1": 300.0,
}

ACTION_COST_REMOTE    = 1.0   # SCADA click
ACTION_COST_MANUAL    = 5.0   # truck roll
RING_RELOCATE_PENALTY = 2.0   # extra cost to move a healthy ring's open point
RESTORE_WEIGHT        = 100.0 # per customer; dwarfs switching costs


# ===========================================================================
# 1-2. SYNTHETIC YEAR + SEASONAL MAXIMA
# ===========================================================================
def synthesize_year(feeders, peak_targets):
    """Hourly amps for one non-leap year per feeder CB, summer-peaking."""
    hours = 8760
    moh = np.repeat(
        np.arange(1, 13),
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    )
    moh = np.repeat(moh, 24)
    hod = np.tile(np.arange(24), 365)
    profiles = {}
    for f in feeders:
        daily    = 0.65 + 0.35 * np.exp(-((hod - 17.5) ** 2) / 18.0)
        seasonal = 0.75 + 0.25 * np.cos((moh - 7) * np.pi / 6)
        raw = daily * seasonal * RNG.normal(1.0, 0.04, hours)
        profiles[f] = raw * (peak_targets[f] / raw[moh == 7].max())
    return profiles, moh


def seasonal_max(profiles, moh, season):
    mask = np.isin(moh, SEASON_MONTHS[season])
    return {f: float(p[mask].max()) for f, p in profiles.items()}


# ===========================================================================
# 3. TREE-AWARE LOAD ALLOCATION
# ===========================================================================
def normal_radial_tree(feeder):
    """BFS tree of the feeder under its present normally-radial switch state."""
    root = FEEDERS[feeder]["root"]
    G = nx.Graph()
    G.add_node(root)
    for ed in EDGES.values():
        if ed["closed"] != 1:
            continue
        fn_u = NODES.get(ed["u"], {}).get("feeder")
        fn_v = NODES.get(ed["v"], {}).get("feeder")
        if fn_u == feeder and fn_v == feeder:
            G.add_edge(ed["u"], ed["v"])
    return nx.bfs_tree(G, root) if root in G else G


def allocate_block_loads(season_peaks):
    """Push each feeder CB peak down the tree, anchor RTU readings where
    present, then spread the residual by transformer kVA over unmeasured blocks.

    Each RTU entry fixes the load for its downstream block set; the remaining
    load is distributed purely by kVA across the unmeasured blocks.
    """
    block_amps = {}
    for fname in FEEDERS:
        total  = season_peaks[fname]
        blocks = [n for n in NODES if NODES[n].get("feeder") == fname]
        kva    = {b: NODES[b]["kva"] for b in blocks}

        measured = {}
        for rtu in FEEDER_RTUS.get(fname, []):
            ds       = rtu["downstream"]
            ds_frac  = sum(kva[b] for b in ds) / sum(kva.values())
            reading  = min(total, total * ds_frac * RNG.normal(1.0, 0.02))
            wsum     = sum(kva[b] for b in ds)
            for b in ds:
                measured[b] = reading * kva[b] / wsum

        residual   = total - sum(measured.values())
        unmeasured = [b for b in blocks if b not in measured]
        wsum       = sum(kva[b] for b in unmeasured) or 1.0
        for b in unmeasured:
            block_amps[b] = max(0.0, residual) * kva[b] / wsum
        block_amps.update(measured)
    return block_amps


# ===========================================================================
# 4. MILP — connectivity-flow radiality (necessary AND sufficient)
# ===========================================================================
def solve_offload(block_amps, outage_sub, verbose=False):
    """Maximise restored customers subject to:
      - outage substation CBs forced open, injection = 0
      - KCL power flow on every live node
      - thermal limits on every closed edge
      - single-commodity connectivity flow proves every live block reachable
        from a healthy source on the closed subgraph (radiality)
      - forest edge count: |closed edges| = |live nodes| - |healthy buses|
      - switching effort penalty (remote < manual; ring relocation +extra)
    """
    prob = pulp.LpProblem("offload_v2", pulp.LpMaximize)

    nodes, edges  = list(NODES), list(EDGES)
    buses         = [n for n, d in NODES.items() if d["kind"] == "bus"]
    healthy_buses = [b for b in buses if b != outage_sub]

    outage_cbs = [e for e, ed in EDGES.items()
                  if ed["kind"] == "cb" and outage_sub in (ed["u"], ed["v"])]
    _cb_ends   = [ed["v"] if ed["u"] == outage_sub else ed["u"]
                  for ed in (EDGES[e] for e in outage_cbs)]
    outage_feeders = {NODES[n]["feeder"] for n in _cb_ends}
    outage_blocks  = [n for n, d in NODES.items()
                      if d["kind"] == "block" and d["feeder"] in outage_feeders]

    demand = {n: block_amps.get(n, 0.0) for n in nodes}
    Ncap   = len(nodes)   # connectivity-flow capacity upper bound

    # --- decision variables --------------------------------------------------
    x  = {e: pulp.LpVariable(f"x_{e}",  cat="Binary")                       for e in edges}
    y  = {n: pulp.LpVariable(f"y_{n}",  cat="Binary")                       for n in nodes}
    fp = {e: pulp.LpVariable(f"fp_{e}", lowBound=-EDGES[e]["amp"],
                              upBound=EDGES[e]["amp"])                        for e in edges}
    g  = {b: pulp.LpVariable(f"g_{b}",  lowBound=0,
                              upBound=NODES[b]["headroom_amps"])              for b in buses}
    a  = {e: pulp.LpVariable(f"a_{e}",  lowBound=0)                         for e in edges}
    cf = {e: pulp.LpVariable(f"cf_{e}", lowBound=-Ncap, upBound=Ncap)       for e in edges}

    # --- fixed states --------------------------------------------------------
    for e in outage_cbs:
        prob += x[e] == 0
    prob += g[outage_sub] == 0
    prob += y[outage_sub] == 0
    for b in healthy_buses:
        prob += y[b] == 1
    for n, d in NODES.items():
        if d["kind"] == "block" and n not in outage_blocks:
            prob += y[n] == 1           # healthy customers stay energised
    for e, ed in EDGES.items():
        if not ed["switchable"]:
            prob += x[e] == ed["closed"]

    # --- power-flow KCL + thermal + energisation coupling -------------------
    for n in nodes:
        inflow = (pulp.lpSum(fp[e] for e, ed in EDGES.items() if ed["v"] == n)
                - pulp.lpSum(fp[e] for e, ed in EDGES.items() if ed["u"] == n))
        inj = g[n] if n in buses else 0
        prob += inflow + inj == demand[n] * y[n]
    for e, ed in EDGES.items():
        prob += fp[e] <=  ed["amp"] * x[e]
        prob += fp[e] >= -ed["amp"] * x[e]
        prob += x[e] <= y[ed["u"]]
        prob += x[e] <= y[ed["v"]]

    # --- connectivity-flow radiality ----------------------------------------
    # Each live non-source node absorbs exactly 1 connectivity unit; sources
    # may supply up to Ncap units each.  Flow is only permitted on closed edges.
    # A feasible single-commodity flow exists iff every live node is reachable
    # from a source on the closed subgraph; combined with the tree-edge-count
    # constraint this forces a spanning forest (necessary AND sufficient).
    for n in nodes:
        cin = (pulp.lpSum(cf[e] for e, ed in EDGES.items() if ed["v"] == n)
             - pulp.lpSum(cf[e] for e, ed in EDGES.items() if ed["u"] == n))
        if n in buses:
            prob += cin >= -Ncap * y[n]
            prob += cin <= 0
        else:
            prob += cin == y[n]         # each live block absorbs 1 unit
    for e, ed in EDGES.items():
        prob += cf[e] <=  Ncap * x[e]
        prob += cf[e] >= -Ncap * x[e]
    prob += (pulp.lpSum(x.values())
             == pulp.lpSum(y.values()) - len(healthy_buses))

    # --- switching effort ----------------------------------------------------
    cost = {}
    for e, ed in EDGES.items():
        if e in outage_cbs or not ed["switchable"]:
            cost[e] = 0.0
            prob += a[e] == 0
            continue
        c = ACTION_COST_REMOTE if ed["remote"] else ACTION_COST_MANUAL
        if ed["kind"] == "ring":
            c += RING_RELOCATE_PENALTY
        cost[e] = c
        prob += a[e] >= x[e] - ed["closed"]
        prob += a[e] >= ed["closed"] - x[e]

    prob += (RESTORE_WEIGHT * pulp.lpSum(NODES[n]["customers"] * y[n] for n in outage_blocks)
             - pulp.lpSum(cost[e] * a[e] for e in edges))

    st = prob.solve(pulp.PULP_CBC_CMD(msg=verbose))
    if pulp.LpStatus[st] != "Optimal":
        raise RuntimeError(f"solver: {pulp.LpStatus[st]}")

    return dict(
        x={e: int(round(x[e].value())) for e in edges},
        y={n: int(round(y[n].value())) for n in nodes},
        outage_cbs=outage_cbs,
        outage_blocks=outage_blocks,
        healthy_buses=healthy_buses,
    )


# ===========================================================================
# 5. PLAN EXTRACTION, PAIRING, VERIFY
# ===========================================================================
def pickup_for_tie(t, sol, G, block_amps):
    """Blocks picked up by closing tie t, their source bus, and estimated load."""
    ed          = EDGES[t]
    out_end     = ed["v"] if ed["v"] in sol["outage_blocks"] else ed["u"]
    healthy_end = ed["u"] if out_end == ed["v"] else ed["v"]
    H      = G.copy()
    H.remove_edge(ed["u"], ed["v"])
    island = nx.node_connected_component(H, out_end)
    blocks = sorted(b for b in island if b in sol["outage_blocks"])
    src    = next(b for b in sol["healthy_buses"] if nx.has_path(G, b, healthy_end))
    return dict(
        blocks=blocks,
        source=src,
        via_feeder=NODES[healthy_end].get("feeder", "?"),
        amps=sum(block_amps[b] for b in blocks),
    )


def build_plan(sol):
    """Return (opens, closes, final_graph) from a solver solution."""
    opens, closes = [], []
    for e, ed in EDGES.items():
        if e in sol["outage_cbs"] or not ed["switchable"]:
            continue
        if sol["x"][e] != ed["closed"]:
            (closes if sol["x"][e] == 1 else opens).append(e)
    G = nx.Graph()
    G.add_nodes_from(n for n in NODES if sol["y"][n])
    for e, ed in EDGES.items():
        if sol["x"][e]:
            G.add_edge(ed["u"], ed["v"], id=e)
    return opens, closes, G


def verify(sol, block_amps, G):
    """Independent checks on the final switched state."""
    report = []

    report.append(("outage substation CBs open",
                   all(sol["x"][e] == 0 for e in sol["outage_cbs"])))
    report.append(("final topology is radial (forest)", nx.is_forest(G)))
    report.append(("no component ties two substations",
                   all(sum(NODES[n]["kind"] == "bus" for n in c) <= 1
                       for c in nx.connected_components(G))))

    # Re-derive edge currents from the BFS tree and verify thermal limits.
    loading = {}
    for comp in nx.connected_components(G):
        srcs = [n for n in comp if NODES[n]["kind"] == "bus"]
        if not srcs:
            continue
        T = nx.bfs_tree(G.subgraph(comp), srcs[0])
        for u, v in T.edges():
            eid        = G[u][v]["id"]
            downstream = nx.descendants(T, v) | {v}
            loading[eid] = (sum(block_amps.get(n, 0.0) for n in downstream),
                            EDGES[eid]["amp"])
    report.append(("all sections within thermal rating",
                   all(a <= c + 1e-6 for a, c in loading.values())))

    restored = [b for b in sol["outage_blocks"] if     sol["y"][b]]
    shed     = [b for b in sol["outage_blocks"] if not sol["y"][b]]
    return report, loading, restored, shed


def explain_single_tie_rejections(block_amps):
    """Flag ties whose thermal rating is insufficient to carry their full
    outage-feeder load alone — indicating the planner will need to sectionalize
    (add an OPEN step) before closing that tie."""
    outage_feeder_names = {
        NODES[ed["v"]]["feeder"]
        if NODES.get(ed["v"], {}).get("kind") == "block"
        else NODES[ed["u"]]["feeder"]
        for _, ed in EDGES.items()
        if ed["kind"] == "cb"
        and (ed["u"] == OUTAGE_SUBSTATION or ed["v"] == OUTAGE_SUBSTATION)
    }
    notes = []
    for t, ed in EDGES.items():
        if ed["kind"] != "tie":
            continue
        for end in (ed["u"], ed["v"]):
            nd = NODES.get(end, {})
            if nd.get("kind") == "block" and nd.get("feeder") in outage_feeder_names:
                feeder = nd["feeder"]
                total  = sum(block_amps.get(b, 0.0) for b, d in NODES.items()
                             if d.get("feeder") == feeder)
                if total > ed["amp"]:
                    notes.append(
                        f"closing {t} alone would push {total:.0f} A through a "
                        f"{ed['amp']:.0f} A tie -> a split (extra OPEN) is forced"
                    )
                break
    return notes


# ===========================================================================
# 6. NETWORK DIAGRAM (ADMS-STYLE ONE-LINE VIEW)
# ===========================================================================
def draw_network(sol=None, block_amps=None):
    """ADMS-style one-line diagram with orthogonal routing for the v2 network.

    sol        : result from solve_offload(); None shows the initial outage state.
    block_amps : block-name -> amps; annotates each load block when provided.

    Tie and ring switches are drawn dashed.  Where a dashed tie line must cross
    a feeder segment it does not connect to, a semicircle flyover (hop) is drawn.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        from matplotlib.patches import Arc
    except ImportError:
        raise ImportError(
            "matplotlib is required for draw_network().  pip install matplotlib"
        )

    BG    = "#151525"
    HOP_R = 0.12

    # ── Node positions ───────────────────────────────────────────────────────
    POS = {
        # substations
        "SUB-A": (1.5,  8.5),
        "SUB-B": (5.5,  8.5),
        "SUB-X": (10.5, 8.5),
        # SUB-A feeders
        "A1-a":  (1.5,  7.0),  "A1-b": (1.5,  5.5),
        "A2-a":  (3.0,  7.0),
        # SUB-B feeder
        "B1-a":  (5.5,  7.0),  "B1-b": (5.5,  5.5),
        # X1 ring + tee
        "X1-A":  (9.5,  7.0),
        "X1-B":  (8.5,  5.5),  "X1-C": (8.5,  4.0),   # leg-1
        "X1-D": (10.5,  5.5),  "X1-E": (10.5, 4.0),   # leg-2 tee
        "X1-F":  (9.5,  2.5),                           # ring join node
        # X2 tee
        "X2-A": (12.0,  7.0),
        "X2-B": (12.0,  5.5),  "X2-C": (13.0, 5.5),   # lateral
    }

    FEEDER_COLORS = {
        "A1": "#1a78c2", "A2": "#2e86ab",
        "B1": "#27ae60",
        "X1": "#e07b00", "X2": "#cc3333",
    }

    # ── Orthogonal routes (ordered waypoint lists) ───────────────────────────
    # All connections are horizontal or vertical only.
    ROUTES = {
        # circuit breakers
        "CB-A1":   [(1.5,  8.5), (1.5,  7.0)],
        "CB-A2":   [(1.5,  8.5), (3.0,  8.5), (3.0,  7.0)],
        "CB-B1":   [(5.5,  8.5), (5.5,  7.0)],
        "CB-X1":   [(10.5, 8.5), (9.5,  8.5), (9.5,  7.0)],
        "CB-X2":   [(10.5, 8.5), (12.0, 8.5), (12.0, 7.0)],
        # solid cables (non-switchable)
        "S-A1-ab": [(1.5,  7.0), (1.5,  5.5)],
        "S-B1-ab": [(5.5,  7.0), (5.5,  5.5)],
        "S-X2-BC": [(12.0, 5.5), (13.0, 5.5)],
        # X1 internal switches
        "S-X1-AB": [(9.5,  7.0), (8.5,  7.0), (8.5,  5.5)],   # L -> leg-1
        "S-X1-BC": [(8.5,  5.5), (8.5,  4.0)],
        "S-X1-AD": [(9.5,  7.0), (10.5, 7.0), (10.5, 5.5)],   # L -> leg-2 tee
        "S-X1-DE": [(10.5, 5.5), (10.5, 4.0)],
        "S-X1-EF": [(10.5, 4.0), (10.5, 2.5), (9.5,  2.5)],   # L -> ring join
        "R-X1":    [(8.5,  4.0), (8.5,  2.5), (9.5,  2.5)],   # N/O ring point
        # X2 internal
        "S-X2-AB": [(12.0, 7.0), (12.0, 5.5)],
        # inter-substation ties (normally open, drawn dashed)
        # T1: A1-b down to y=4.0, then across to X1-C
        "T1": [(1.5,  5.5), (1.5,  4.0), (8.5,  4.0)],
        # T2: B1-a right to x=7.0, down to y=4.0, right to X1-E
        "T2": [(5.5,  7.0), (7.0,  7.0), (7.0,  4.0), (10.5, 4.0)],
        # T3: A2-a down to y=5.0, across to x=13.0, up to X2-C
        "T3": [(3.0,  7.0), (3.0,  5.0), (13.0, 5.0), (13.0, 5.5)],
        # T4: B1-b up to y=6.0, across to x=12.0, down to X2-B
        "T4": [(5.5,  5.5), (5.5,  6.0), (12.0, 6.0), (12.0, 5.5)],
    }

    # Where a tie/ring crosses a feeder segment it does not connect to, the
    # tie carries a semicircle flyover at that (x, y) crossing point.
    HOPS = {
        # T2 horizontal at y=4.0 passes over X1-C at x=8.5 (T1 terminus + S-X1-BC)
        "T2": [(8.5, 4.0)],
        # T3 horizontal at y=5.0 is crossed by S-X1-BC (x=8.5) and S-X1-DE (x=10.5)
        "T3": [(8.5, 5.0), (10.5, 5.0)],
        # T4 horizontal at y=6.0 is crossed by S-X1-AB (x=8.5) and S-X1-AD (x=10.5)
        "T4": [(8.5, 6.0), (10.5, 6.0)],
    }

    # Switch-symbol positions (chosen to avoid route corners and node circles).
    SYMBOL_POS = {
        "CB-A1":   (1.5,  7.75),  "CB-A2":   (3.0,  7.75),
        "CB-B1":   (5.5,  7.75),
        "CB-X1":   (9.5,  7.75),  "CB-X2":   (12.0, 7.75),
        "S-X1-AB": (8.5,  6.25),  "S-X1-BC": (8.5,  4.75),
        "S-X1-AD": (10.5, 6.25),  "S-X1-DE": (10.5, 4.75),
        "S-X1-EF": (10.5, 3.25),  "R-X1":    (8.5,  3.25),
        "S-X2-AB": (12.0, 6.25),
        "T1":      (5.0,  4.0),
        "T2":      (7.0,  5.5),
        "T3":      (8.0,  5.0),
        "T4":      (8.75, 6.0),
    }

    # ── Switch / energisation states ─────────────────────────────────────────
    if sol is not None:
        xstate = sol["x"]
        ystate = sol["y"]
    else:
        xstate = {e: ed["closed"] for e, ed in EDGES.items()}
        _ocbs  = {e for e, ed in EDGES.items()
                  if ed["kind"] == "cb"
                  and (ed["u"] == OUTAGE_SUBSTATION or ed["v"] == OUTAGE_SUBSTATION)}
        _of    = {NODES[ed["v"]]["feeder"]
                  for e, ed in EDGES.items()
                  if e in _ocbs and NODES.get(ed["v"], {}).get("kind") == "block"}
        ystate = {
            n: 0 if (n == OUTAGE_SUBSTATION or
                     (NODES[n]["kind"] == "block" and NODES[n].get("feeder") in _of))
               else 1
            for n in NODES
        }

    # ── Canvas ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(1.2, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    def _draw_route(pts, hops_xy, color, lw, ls, zorder):
        """Draw an orthogonal polyline; insert a semicircle flyover at each hop."""
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            is_h   = abs(y1 - y0) < 1e-9
            seg_hops = []
            if is_h:
                for hx, hy in hops_xy:
                    if (abs(hy - y0) < 1e-9 and
                            min(x0, x1) + 1e-9 < hx < max(x0, x1) - 1e-9):
                        seg_hops.append(hx)
            if not seg_hops:
                ax.plot([x0, x1], [y0, y1], color=color, lw=lw,
                        linestyle=ls, solid_capstyle="butt", zorder=zorder)
            else:
                seg_hops.sort(reverse=(x0 > x1))
                cur = x0
                for hx in seg_hops:
                    ax.plot([cur, hx - HOP_R], [y0, y0], color=color, lw=lw,
                            linestyle=ls, solid_capstyle="butt", zorder=zorder)
                    ax.add_patch(Arc(
                        (hx, y0), 2 * HOP_R, 2 * HOP_R,
                        angle=0, theta1=0, theta2=180,
                        color=color, lw=lw, zorder=zorder + 1,
                    ))
                    cur = hx + HOP_R
                ax.plot([cur, x1], [y0, y1], color=color, lw=lw,
                        linestyle=ls, solid_capstyle="butt", zorder=zorder)

    # ── Draw edges ────────────────────────────────────────────────────────────
    for eid, ed in EDGES.items():
        is_closed  = bool(xstate[eid])
        feeder     = NODES[ed["u"]].get("feeder") or NODES[ed["v"]].get("feeder")
        line_color = FEEDER_COLORS.get(feeder, "#aaaaaa") if is_closed else "#445566"
        lw = 3.0 if ed["kind"] in ("cb", "cable") else 2.0
        ls = "--" if ed["kind"] in ("tie", "ring") else "-"

        _draw_route(ROUTES[eid], HOPS.get(eid, []),
                    line_color, lw, ls, zorder=2)

        if ed["switchable"]:
            sx, sy    = SYMBOL_POS[eid]
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
            pts = ROUTES[eid]
            mx  = (pts[0][0] + pts[-1][0]) / 2
            my  = (pts[0][1] + pts[-1][1]) / 2
            ax.text(mx + 0.18, my, eid,
                    fontsize=6.5, ha="left", va="center", color="#888899",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=BG,
                              edgecolor="none", alpha=0.65),
                    zorder=7)

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for nid, nd in NODES.items():
        px, py    = POS[nid]
        energized = bool(ystate.get(nid, 1))

        if nd["kind"] == "bus":
            fc = "#c0392b" if nid == OUTAGE_SUBSTATION else "#2471a3"
            if not energized:
                fc = "#3a3a4a"
            ax.add_patch(mpatches.FancyBboxPatch(
                (px - 0.52, py - 0.21), 1.04, 0.42,
                boxstyle="round,pad=0.06",
                facecolor=fc, edgecolor="#888899", lw=1.5, zorder=8,
            ))
            tag = " [OFF]" if not energized else ""
            ax.text(px, py, nid + tag, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=9)
        else:
            fc = FEEDER_COLORS.get(nd["feeder"], "#666") if energized else "#2e2e3e"
            ax.add_patch(plt.Circle(
                (px, py), 0.34, facecolor=fc, edgecolor="#888899", lw=1.5, zorder=8
            ))
            ax.text(px, py + 0.08, nid, ha="center", va="center",
                    fontsize=6.5, fontweight="bold",
                    color="white" if energized else "#666677", zorder=9)
            ax.text(px, py - 0.12, f"{nd['customers']}c",
                    ha="center", va="center", fontsize=6,
                    color="white" if energized else "#666677", zorder=9)
            if block_amps and nid in block_amps:
                ax.text(px + 0.42, py + 0.25, f"{block_amps[nid]:.0f} A",
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
                      label="Tie / ring switch (dashed)"),
        mlines.Line2D([0], [0], color="w", marker="o", markerfacecolor=BG,
                      markeredgecolor="#dddddd", markersize=12, markeredgewidth=2.5,
                      label="Switch closed (●)"),
        mlines.Line2D([0], [0], color="w", marker="o", markerfacecolor=BG,
                      markeredgecolor="#ff6b6b", markersize=12, markeredgewidth=2.5,
                      label="Switch open (×)"),
        mlines.Line2D([0], [0], color="#888899", lw=2,
                      label="⌢  crossing — not a junction"),
    ]
    leg = ax.legend(handles=leg_items, loc="lower left", fontsize=8,
                    facecolor="#1e1e30", edgecolor="#444455", labelcolor="white",
                    title="Legend", title_fontsize=8)
    leg.get_title().set_color("#ccccdd")

    mode = "Post-switching state" if sol else "Initial outage state"
    ax.set_title(
        f"Substation Outage Offload Planner v2  ·  {mode}",
        fontsize=13, fontweight="bold", color="white", pad=14,
    )
    plt.tight_layout()
    plt.show()
    return fig, ax


# ===========================================================================
# MAIN
# ===========================================================================
def main(season="summer"):
    profiles, moh = synthesize_year(FEEDERS, FEEDER_PEAK_TARGET)
    peaks         = seasonal_max(profiles, moh, season)
    block_amps    = allocate_block_loads(peaks)

    print(f"=== SUBSTATION OUTAGE OFFLOAD PLAN v2  (season: {season}) ===")
    print(f"Outage: {OUTAGE_SUBSTATION}  (CB-X1 ring+tee feeder, CB-X2 tee feeder)\n")

    print("Seasonal feeder maxima (from 1 yr of synthetic CB data):")
    for fn, v in peaks.items():
        print(f"  {fn}: {v:6.1f} A")

    print("\nAllocated block loads (kVA-weighted, RTU-anchored where present):")
    for b, v in sorted(block_amps.items()):
        fd  = NODES[b]["feeder"]
        tag = ""
        for rtu in FEEDER_RTUS.get(fd, []):
            if b in rtu["downstream"]:
                tag = f"  [below RTU {rtu['switch']}]"
        print(f"  {b}: {v:6.1f} A{tag}")

    sol = solve_offload(block_amps, OUTAGE_SUBSTATION)
    opens, closes, G = build_plan(sol)

    for note in explain_single_tie_rejections(block_amps):
        print(f"\nNote: {note}")

    print("\n--- SWITCHING SEQUENCE ---")
    step = 0
    for e in sol["outage_cbs"]:
        step += 1
        print(f"Step {step}: VERIFY OPEN  {e}  (feeder breaker, substation de-energised)")
    for e in opens:
        step += 1
        ed   = EDGES[e]
        mode = "remote" if ed["remote"] else "MANUAL"
        kind = "ring open-point relocation" if ed["kind"] == "ring" else "sectionalize"
        print(f"Step {step}: OPEN   {e:9s}  ({mode})  {kind} {ed['u']} | {ed['v']}")
    for e in closes:
        step += 1
        ed   = EDGES[e]
        mode = "remote" if ed["remote"] else "MANUAL"
        p    = pickup_for_tie(e, sol, G, block_amps) if ed["kind"] == "tie" else None
        if p:
            print(f"Step {step}: CLOSE  {e:9s}  ({mode})  backfeed {', '.join(p['blocks'])} "
                  f"from {p['source']} via {p['via_feeder']}  (~{p['amps']:.0f} A)")
        else:
            print(f"Step {step}: CLOSE  {e:9s}  ({mode})  close {ed['u']} | {ed['v']}")

    print("\n--- OPEN/CLOSE PAIRING (radial integrity) ---")
    for e in closes:
        if EDGES[e]["kind"] == "tie":
            p        = pickup_for_tie(e, sol, G, block_amps)
            boundary = [o for o in opens + sol["outage_cbs"]
                        if EDGES[o]["u"] in p["blocks"] or EDGES[o]["v"] in p["blocks"]]
            print(f"  CLOSE {e} is bounded by OPEN "
                  f"{', '.join(boundary) or '(none – full feeder segment)'}")

    report, loading, restored, shed = verify(sol, block_amps, G)
    print("\n--- VERIFICATION ---")
    for name, ok in report:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    ct = sum(NODES[b]["customers"] for b in sol["outage_blocks"])
    cr = sum(NODES[b]["customers"] for b in restored)
    print(f"\nRestored {len(restored)}/{len(sol['outage_blocks'])} blocks, "
          f"{cr}/{ct} customers ({100*cr/ct:.0f}%)")
    if shed:
        print(f"Unrestorable (shed): {', '.join(shed)}")

    print("\nPost-switching section loading:")
    for e, (amps, cap) in sorted(loading.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print(f"  {e:9s}  {amps:6.1f} / {cap:.0f} A  ({100*amps/cap:5.1f}%)")

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
