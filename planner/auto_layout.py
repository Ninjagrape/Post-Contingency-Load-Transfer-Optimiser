"""
auto_layout.py
==============
Derive an ADMS-style one-line layout (pos / routes / hops / symbol_pos) purely
from network topology (nodes + edges), so a scenario JSON need only describe
*connectivity*, not geometry.

Strategy (layered / Sugiyama-flavoured, tuned for distribution feeders):
  * The network minus its loop-closing edges (ties + rings) is a forest: every
    substation bus plus the feeders hanging off it is one tree.
  * Each tree is laid out tidily: depth from the bus sets the row (y), a
    left-to-right leaf packing sets the column (x).  Trees are packed side by
    side.  This reproduces the "sources on a top rail, feeders descending in
    columns" house style without any hand placement.
  * Every edge attaches to a *port*: a distinct point on a node's perimeter
    (top / bottom / left / right, fanned when a side carries several edges).
    Routes are drawn port-to-port, so two wires reaching the same node never
    leave from the same spot.
  * Tie / ring edges are loop-closers.  Each gets its own horizontal *lane*
    beneath the diagram and drops to it at its (per-edge-unique) port x, so
    parallel ties into one node run side by side instead of on top of each
    other.  Crossings against feeder segments become semicircle hops.

Returns a dict with the exact keys draw_network() already consumes:
  pos, routes, hops, symbol_pos, feeder_colors, figsize, xlim, ylim
"""

import networkx as nx

# ---- tunables --------------------------------------------------------------
TOP_Y       = 0.0     # bus row (diagram grows downward into negative y, flipped at the end)
LAYER_GAP   = 2.8     # vertical gap between tree depths
LEAF_GAP    = 2.4     # horizontal gap between adjacent leaf columns
TREE_GAP    = 3.5     # horizontal gap between separate substation trees
NODE_R      = 0.34    # node radius (matches draw_network)
PORT_SPREAD = 0.55    # how far across a side fanned ports spread (fraction of 2R)
LANE_GAP    = 1.2     # vertical gap between tie lanes
LANE_TOP_PAD= 2.0     # gap between lowest node and the first tie lane
CORRIDOR_CLEAR = NODE_R * 0.9   # min x-gap before a tie drop counts as overlapping
CORRIDOR_STEP  = NODE_R * 1.5   # sideways nudge when a tie drop must dodge a wire
HOP_EPS     = 1e-6

_PALETTE = ["#1a78c2", "#27ae60", "#9b59b6", "#e67e22", "#16a085",
            "#c0a020", "#2e9bd6", "#d65f9a", "#e8743b", "#5d6d7e"]
_OUTAGE_PALETTE = ["#e0b000", "#cc3333", "#d65f9a", "#e8743b", "#cc8800"]


# ===========================================================================
# helpers
# ===========================================================================
def _loop_edge(ed):
    return ed["kind"] in ("tie", "ring")


def _forest(nodes, edges):
    """Graph of all NON-loop edges (cb/switch/cable): the normal radial forest."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for eid, ed in edges.items():
        if not _loop_edge(ed):
            G.add_edge(ed["u"], ed["v"], id=eid)
    return G


def _tidy_tree(G, root, depth):
    """Assign integer leaf columns under `root`, centre parents over children.

    Returns {node: column_float}.  Pure tree assumed (forest component).
    """
    col = {}
    counter = [0]
    order = sorted(G[root])  # deterministic child order

    def rec(n, parent):
        kids = [k for k in sorted(G[n]) if k != parent]
        if not kids:
            col[n] = counter[0]
            counter[0] += 1
            return col[n]
        cs = [rec(k, n) for k in kids]
        col[n] = sum(cs) / len(cs)
        return col[n]

    rec(root, None)
    return col


def _classify_ports(node, incident, pos, depth):
    """Decide which side each incident edge leaves `node` from.

    incident: list of (edge_id, other_node, is_loop)
    Returns {edge_id: side} with side in {'top','bottom','left','right'}.
    """
    side = {}
    cx, cy = pos[node]
    for eid, other, is_loop in incident:
        ox, oy = pos[other]
        if not is_loop:
            # tree edge: parent (shallower) leaves the top, children the bottom
            side[eid] = "top" if depth[other] < depth[node] else "bottom"
        else:
            # loop edge: leave toward whichever side the partner sits on
            side[eid] = "left" if ox < cx - HOP_EPS else "right"
    return side


def _fan_ports(node, incident, side, pos):
    """Spread the edges assigned to each side across that side, return
    {edge_id: (port_x, port_y)} on the node perimeter."""
    cx, cy = pos[node]
    by_side = {}
    for eid, other, is_loop in incident:
        by_side.setdefault(side[(eid, node)], []).append((eid, other))
    port = {}
    for s, items in by_side.items():
        # order items along the side so wires don't cross at the node
        if s in ("top", "bottom"):
            items.sort(key=lambda io: pos[io[1]][0])      # by partner x
        else:
            items.sort(key=lambda io: -pos[io[1]][1])     # by partner y (down first)
        m = len(items)
        for i, (eid, _other) in enumerate(items):
            t = 0.0 if m == 1 else (i / (m - 1) - 0.5)     # -0.5..0.5
            off = t * 2 * NODE_R * PORT_SPREAD
            if s == "top":
                port[eid] = (cx + off, cy + NODE_R)
            elif s == "bottom":
                port[eid] = (cx + off, cy - NODE_R)
            elif s == "left":
                port[eid] = (cx - NODE_R, cy + off)
            else:  # right
                port[eid] = (cx + NODE_R, cy + off)
    return port


def _ortho(p0, p1):
    """Two-segment Manhattan route between two ports (horizontal then vertical,
    via an elbow).  Near-aligned ports snap to a single straight run so a
    stacked tree edge doesn't kink.

    When p0 is above p1 (the normal parent→child case): jog horizontally to
    p1's x first, then drop straight down.  This puts the vertical segment at
    the *child's* x so the wire arrives at the child's centre-top rather than
    approaching diagonally from the parent's position."""
    (x0, y0), (x1, y1) = p0, p1
    SNAP = 2 * NODE_R * PORT_SPREAD + HOP_EPS   # within a port-fan's spread
    if abs(x0 - x1) < SNAP:
        xm = (x0 + x1) / 2.0
        return [(xm, y0), (xm, y1)]
    if abs(y0 - y1) < SNAP:
        ym = (y0 + y1) / 2.0
        return [(x0, ym), (x1, ym)]
    if y0 > y1:
        # p0 is higher (closer to bus): jog to child x, then drop straight down
        return [(x0, y0), (x1, y0), (x1, y1)]
    # p0 is lower: drop first, then jog (reversed/upward route)
    return [(x0, y0), (x0, y1), (x1, y1)]


def _seg_crosses(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    """True if horizontal segment a crosses vertical segment b (interiors)."""
    # a horizontal at y=ay0 spanning x in [ax0,ax1]; b vertical at x=bx0 spanning y
    if abs(ay0 - ay1) > HOP_EPS or abs(bx0 - bx1) > HOP_EPS:
        return None
    y = ay0; x = bx0
    if min(ax0, ax1) + HOP_EPS < x < max(ax0, ax1) - HOP_EPS and \
       min(by0, by1) + HOP_EPS < y < max(by0, by1) - HOP_EPS:
        return (x, y)
    return None


# ===========================================================================
# main entry
# ===========================================================================
def auto_layout(nodes, edges, feeders, outage_substation, base_diagram=None):
    base_diagram = base_diagram or {}
    forced_pos = {k: tuple(v) for k, v in base_diagram.get("pos", {}).items()}

    G = _forest(nodes, edges)

    # --- depth from nearest bus (BFS over the forest) -----------------------
    buses = [n for n, d in nodes.items() if d["kind"] == "bus"]
    depth = {}
    roots = []
    for b in buses:
        if b in depth:
            continue
        depth[b] = 0
        roots.append(b)
        for n, d in nx.single_source_shortest_path_length(G, b).items():
            depth[n] = d
    for n in nodes:                       # isolated safety
        depth.setdefault(n, 0)

    # --- per-tree tidy columns, packed left to right ------------------------
    pos = {}
    cursor = 0.0
    for b in sorted(roots, key=lambda r: r):
        comp = nx.node_connected_component(G, b)
        col = _tidy_tree(G.subgraph(comp), b, depth)
        cmin = min(col.values())
        span = max(col.values()) - cmin
        for n, c in col.items():
            x = cursor + (c - cmin) * LEAF_GAP
            y = TOP_Y - depth[n] * LAYER_GAP
            pos[n] = (x, y)
        cursor += span * LEAF_GAP + TREE_GAP
    pos.update(forced_pos)               # honour any pinned nodes

    # --- collect incident edges per node ------------------------------------
    incident = {n: [] for n in nodes}
    for eid, ed in edges.items():
        il = _loop_edge(ed)
        incident[ed["u"]].append((eid, ed["v"], il))
        incident[ed["v"]].append((eid, ed["u"], il))

    # --- assign ports -------------------------------------------------------
    side = {}                            # (edge_id, node) -> side
    for n in nodes:
        for eid, s in _classify_ports(n, incident[n], pos, depth).items():
            side[(eid, n)] = s
    port = {}                            # (edge_id, node) -> (x, y)
    for n in nodes:
        pp = _fan_ports(n, incident[n], side, pos)
        for eid, xy in pp.items():
            port[(eid, n)] = xy

    # --- routes: tree edges first -------------------------------------------
    routes = {}
    for eid, ed in edges.items():
        if _loop_edge(ed):
            continue
        routes[eid] = _ortho(port[(eid, ed["u"])], port[(eid, ed["v"])])

    # --- routes: loop edges (ties/rings) in dedicated lanes -----------------
    # Lanes live just below the deepest tie endpoint, not below the whole tree,
    # so a long stub feeder doesn't push every tie far off the bottom.
    loop_ids = [e for e, ed in edges.items() if _loop_edge(ed)]
    if loop_ids:
        deepest_tie_y = min(min(pos[edges[e]["u"]][1], pos[edges[e]["v"]][1])
                            for e in loop_ids)
    else:
        deepest_tie_y = min((y for _, y in pos.values()), default=0.0)
    # order lanes by horizontal span so short ties sit nearest the diagram
    def _span(eid):
        ed = edges[eid]
        return abs(pos[ed["u"]][0] - pos[ed["v"]][0])
    loop_ids.sort(key=_span)

    # Corridor x for each loop-edge endpoint = the x of its vertical drop down
    # to the tie lane.  We *want* this to sit on the node's centre (clean
    # centre-bottom exit, ADMS house style), but a centred drop collinearly
    # overlaps any wire that already runs in that column: the feeder segment
    # continuing below the node (e.g. T2 off X1-E running down through S-X1-EF),
    # or another tie dropping down the same stacked column (T2 and T4 both down
    # the B1-a/B1-b column).  So we resolve collisions: start at centre and, if
    # the drop would overlap an existing vertical (feeder route or an
    # already-placed tie drop) along a shared x, nudge it sideways - preferring
    # the side its port is on so the short jog doesn't cut back across the node.
    lane_y = {}
    for i, eid in enumerate(loop_ids):
        lane_y[eid] = deepest_tie_y - LANE_TOP_PAD - i * LANE_GAP

    # Obstacle verticals = every vertical segment of the radial (tree) routes,
    # which are the only routes built so far.
    obstacles = []                         # (x, ylo, yhi)
    for pts in routes.values():
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            if abs(x0 - x1) < HOP_EPS and abs(y0 - y1) > HOP_EPS:
                obstacles.append((x0, min(y0, y1), max(y0, y1)))

    placed = []                            # tie drop verticals already routed
    def _clear(x, ylo, yhi):
        for ox, oy0, oy1 in obstacles:
            if abs(ox - x) < CORRIDOR_CLEAR and min(yhi, oy1) - max(ylo, oy0) > HOP_EPS:
                return False
        for ox, oy0, oy1 in placed:
            if abs(ox - x) < CORRIDOR_CLEAR and min(yhi, oy1) - max(ylo, oy0) > HOP_EPS:
                return False
        return True

    def _corridor_x(center, ylo, yhi, prefer):
        if _clear(center, ylo, yhi):
            return center
        for k in range(1, 80):
            for s in (prefer, -prefer):
                x = center + s * k * CORRIDOR_STEP
                if _clear(x, ylo, yhi):
                    return x
        return center                      # give up: better than an exception

    corridor = {}                          # (eid, node) -> drop_x
    for eid in loop_ids:
        ed = edges[eid]
        ly = lane_y[eid]
        pu = port[(eid, ed["u"])]
        pv = port[(eid, ed["v"])]
        uy, vy = pos[ed["u"]][1], pos[ed["v"]][1]
        for node, pp, ny in ((ed["u"], pu, uy), (ed["v"], pv, vy)):
            cx = pos[node][0]
            prefer = 1 if pp[0] >= cx else -1
            ylo, yhi = min(ly, ny), max(ly, ny)
            dx = _corridor_x(cx, ylo, yhi, prefer)
            corridor[(eid, node)] = dx
            placed.append((dx, ylo, yhi))
        dux = corridor[(eid, ed["u"])]
        dvx = corridor[(eid, ed["v"])]
        # port -> short jog to corridor x -> drop -> lane -> drop -> jog -> port
        routes[eid] = [pu, (pu[0], uy), (dux, uy), (dux, ly),
                       (dvx, ly), (dvx, vy), (pv[0], vy), pv]

    # --- hops: where a loop lane's horizontal run crosses a vertical wire ---
    hops = {}
    vert_segs = []                       # all vertical segments from every route
    for eid, pts in routes.items():
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            if abs(x0 - x1) < HOP_EPS and abs(y0 - y1) > HOP_EPS:
                vert_segs.append((eid, x0, y0, y1))
    for eid in loop_ids:
        pts = routes[eid]
        hop_pts = []
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            if abs(y0 - y1) > HOP_EPS:            # only horizontal runs hop
                continue
            for oeid, vx, vy0, vy1 in vert_segs:
                if oeid == eid:
                    continue
                c = _seg_crosses(x0, y0, x1, y1, vx, vy0, vx, vy1)
                if c:
                    hop_pts.append(c)
        if hop_pts:
            hops[eid] = hop_pts

    # --- tee points: junction dots where a feeder branches mid-wire ----------
    # A branch at a *node* (e.g. X1-A feeding both X1-B and X1-D) needs no dot:
    # the node circle already shows the connection, and a dot at the node centre
    # just stacks on top of it.  With port-fan routing every branch emanates from
    # its node, so no off-node junctions arise here and tee_pos stays empty.  The
    # key is kept so draw_network can still render genuine mid-wire tees if a
    # future layout produces them.
    tee_pos = {}

    # --- symbol_pos: midpoint of each switchable edge's route ---------------
    def _polyline_midpoint(pts):
        segs = [((pts[i], pts[i + 1]),
                 ((pts[i + 1][0] - pts[i][0]) ** 2 +
                  (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5)
                for i in range(len(pts) - 1)]
        total = sum(L for _, L in segs)
        if total < HOP_EPS:
            return pts[0]
        half, run = total / 2.0, 0.0
        for (a, b), L in segs:
            if run + L >= half:
                f = (half - run) / L if L > HOP_EPS else 0.0
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
            run += L
        return pts[-1]

    symbol_pos = {}
    for eid, ed in edges.items():
        if not ed.get("switchable"):
            continue
        symbol_pos[eid] = tuple(float(c) for c in _polyline_midpoint(routes[eid]))

    # --- shift y so the whole diagram sits in positive coordinates ----------
    # Buses are at TOP_Y (0) and the tree grows DOWN into negative y, with tie
    # lanes lowest of all.  We only need a constant shift so the lowest point
    # sits just above the axis floor; buses stay highest (no flip needed).
    ally = [p[1] for p in pos.values()] + [p[1] for r in routes.values() for p in r]
    y_lo = min(ally)
    shift = lambda x, y: (x, y - y_lo)
    pos = {n: shift(*p) for n, p in pos.items()}
    routes = {e: [shift(*p) for p in r] for e, r in routes.items()}
    hops = {e: [shift(*p) for p in pts] for e, pts in hops.items()}
    symbol_pos = {e: shift(*p) for e, p in symbol_pos.items()}

    xs = [p[0] for p in pos.values()] + [p[0] for r in routes.values() for p in r]
    ys = [p[1] for p in pos.values()] + [p[1] for r in routes.values() for p in r]
    pad = 2.0
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)
    figsize = (max(20, (xlim[1] - xlim[0]) * 1.4),
               max(12, (ylim[1] - ylim[0]) * 1.4))

    # --- feeder colours (reuse scenario's if given, else auto) --------------
    fcolors = dict(base_diagram.get("feeder_colors", {}))
    outage_feeders, healthy_feeders = set(), set()
    for fn, fd in feeders.items():
        cb = edges.get(fd["cb"], {})
        if outage_substation in (cb.get("u"), cb.get("v")):
            outage_feeders.add(fn)
        else:
            healthy_feeders.add(fn)
    for i, fn in enumerate(sorted(healthy_feeders)):
        fcolors.setdefault(fn, _PALETTE[i % len(_PALETTE)])
    for i, fn in enumerate(sorted(outage_feeders)):
        fcolors.setdefault(fn, _OUTAGE_PALETTE[i % len(_OUTAGE_PALETTE)])

    tee_pos = {n: shift(*p) for n, p in tee_pos.items()}
    
    return dict(pos=pos, routes=routes, hops=hops, symbol_pos=symbol_pos,
                tee_pos=tee_pos, feeder_colors=fcolors, figsize=list(figsize),
                xlim=list(xlim), ylim=list(ylim))
