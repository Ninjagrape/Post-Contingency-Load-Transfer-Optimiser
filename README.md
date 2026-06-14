# auto-post-contingency-offload

A tool that automatically generates optimal switching sequences to restore power after a full substation outage, using mixed-integer linear programming (MILP).

Handles complex real-world topologies: multiple circuit breakers per substation, branching/tee feeders, intra-feeder ring configurations with relocatable normally-open points, and multiple tie paths per outage feeder.

## Overview

When a substation goes offline, operators must manually determine which tie switches to close and which sectionalizing switches to open to backfeed de-energized customers from neighboring substations — without overloading cables or circuit breakers, and without creating network loops.

This tool automates that process:

1. **Synthesizes** a year of hourly circuit breaker load data (stand-in for SCADA history)
2. **Computes** seasonal peak loads per feeder
3. **Allocates** feeder-head current down to individual load blocks using a tree-aware approach: RTU-measured branch flows are anchored first; the remaining load is spread by transformer kVA across unmeasured blocks only
4. **Solves a MILP** to maximize restored customers while penalizing switching actions and enforcing thermal limits and connectivity-flow radiality (necessary and sufficient, handles meshed topologies that defeat a simple edge-count constraint)
5. **Emits** an ordered switching plan with independent state verification and single-tie overload warnings
6. **Renders** an ADMS-style one-line network diagram showing switch states and load before and after restoration

## Usage

```bash
python offload_planner.py [season]
```

`season` is optional and defaults to `summer`. Valid values: `winter`, `spring`, `summer`, `fall`.

### Example output

```
=== SUBSTATION OUTAGE OFFLOAD PLAN v2  (season: summer) ===
Outage: SUB-X  (CB-X1 ring+tee feeder, CB-X2 tee feeder)

Seasonal feeder maxima (from 1 yr of synthetic CB data):
  X1:  360.0 A
  X2:  250.0 A
  ...

Note: closing T2 alone would push 360 A through a 200 A tie -> a split (extra OPEN) is forced

--- SWITCHING SEQUENCE ---
Step 1: VERIFY OPEN  CB-X1  (feeder breaker, substation de-energised)
Step 2: VERIFY OPEN  CB-X2  (feeder breaker, substation de-energised)
Step 3: OPEN   S-X1-BC    (MANUAL)  sectionalize X1-B | X1-C
Step 4: CLOSE  T1         (remote)  backfeed X1-C from SUB-A via A1  (~84 A)
Step 5: CLOSE  T2         (MANUAL)  backfeed X1-D, X1-E, X1-F from SUB-B via B1  (~193 A)
...

--- VERIFICATION ---
  [PASS] outage substation CBs open
  [PASS] final topology is radial (forest)
  [PASS] no component ties two substations
  [PASS] all sections within thermal rating

Restored 9/9 blocks, 1960/1960 customers (100%)
```

## Network model

The built-in network (editable at the top of `offload_planner.py`) represents three substations with a complex mix of feeder topologies:

| Component | Description |
|---|---|
| `SUB-A`, `SUB-B` | Healthy source substations |
| `SUB-X` | The outaged substation (two CBs) |
| `X1` | De-energized ring+tee feeder: branches at X1-A into two legs that rejoin at X1-F via normally-open ring point R-X1 |
| `X2` | De-energized tee feeder: main line X2-A→X2-B with solid-cable lateral to X2-C |
| `A1`, `A2` | Healthy feeders from SUB-A available for backfeed |
| `B1` | Healthy feeder from SUB-B available for backfeed |
| Tie switches `T1–T4` | Normally-open backfeed paths (multiple per outage feeder) |
| Sectionalizing switches | Mid-feeder isolation points (remote or manual) |
| Ring switch `R-X1` | Normally-open ring point; planner may relocate at an extra penalty cost |

### MILP objective and constraints

**Objective:** maximise restored customers (100 pts/customer) minus switching costs:

| Action | Cost |
|---|---|
| Remote switch operation | 1 |
| Manual switch operation (truck roll) | 5 |
| Ring open-point relocation (additional) | +2 |

**Constraints:**

- Kirchhoff's current law at every node
- Thermal ratings on every edge (signed flow variable, both directions)
- **Connectivity-flow radiality**: a single-commodity flow proves every live block reachable from a healthy source on the closed subgraph — necessary *and* sufficient, unlike the simple edge-count test which a meshed topology can defeat
- Tree edge count: `|closed edges| = |live nodes| − |healthy buses|`
- No substation paralleling

## Network diagram

Running the script automatically opens an ADMS-style one-line diagram of the post-switching state. You can also call `draw_network()` directly:

```python
from offload_planner import (draw_network, solve_offload, build_plan,
                              allocate_block_loads, synthesize_year,
                              seasonal_max, FEEDERS, FEEDER_PEAK_TARGET,
                              OUTAGE_SUBSTATION)

profiles, moh = synthesize_year(FEEDERS, FEEDER_PEAK_TARGET)
peaks         = seasonal_max(profiles, moh, "summer")
block_amps    = allocate_block_loads(peaks)
sol           = solve_offload(block_amps, OUTAGE_SUBSTATION)
opens, closes, G = build_plan(sol)

# Initial outage state (X feeders dark, ties open)
draw_network()

# Post-switching restored state with per-block load annotations
draw_network(sol=sol, block_amps=block_amps)
```

The diagram uses a dark canvas with orthogonal routing and feeder color-coding:

| Visual element | Meaning |
|---|---|
| Blue rectangle | Healthy substation bus |
| Red rectangle | Outaged substation (SUB-X) |
| Colored circle | Load block, shaded per feeder; label shows name and customer count |
| Gold annotation | Block load in amps (shown when `block_amps` is provided) |
| Solid feeder-colored line | Energized, closed switch or cable |
| Dark gray line | De-energized section |
| Dashed line | Tie switch or ring switch (normally open) |
| ⌢ semicircle on a dashed line | Flyover crossing — the tie passes over a feeder segment it does not connect to |
| ● filled circle on line | Switch closed |
| × circled cross on line | Switch open |

Requires `matplotlib` (optional — solver and plan output work without it).

## Dependencies

```
numpy
networkx
pulp
matplotlib   # optional, for draw_network()
```

Install with:

```bash
pip install numpy networkx pulp matplotlib
```

PuLP ships with the CBC solver; no external solver installation is required.

## Adapting to a real network

Edit the `NODES`, `EDGES`, `FEEDERS`, and `FEEDER_RTUS` dictionaries at the top of `offload_planner.py`:

- **`NODES`** — add `kind="bus"` entries for each substation and `kind="block"` entries for each load segment between switching devices. Set `kva` to the connected transformer kVA (used as the load-allocation weight).
- **`EDGES`** — define every CB, sectionalizing switch, solid cable, ring switch, and tie switch. Set `switchable=True`/`remote=True` where applicable; solid cables use `switchable=False`.
- **`FEEDERS`** — map each feeder name to its source CB and its root block node (`root`). The topology is derived automatically from the graph, so no ordered block list is needed.
- **`FEEDER_RTUS`** — list any mid-feeder RTU-metered switches and the block names downstream of each. More RTU entries improve load allocation accuracy.
- **`FEEDER_PEAK_TARGET`** — replace with real SCADA historian exports rather than the synthetic year generator.
- **`OUTAGE_SUBSTATION`** — set to the bus node that has gone offline.
