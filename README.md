# auto-post-contingency-offload

A prototype tool that automatically generates optimal switching sequences to restore power after a full substation outage, using mixed-integer linear programming (MILP).

## Overview

When a substation goes offline, operators must manually determine which tie switches to close and which sectionalizing switches to open in order to backfeed the de-energized customers from neighboring substations — without overloading cables or circuit breakers, and without creating network loops.

This tool automates that process:

1. **Synthesizes** a year of hourly circuit breaker load data (stand-in for SCADA history)
2. **Computes** seasonal peak loads per feeder
3. **Allocates** feeder-head current down to individual switchable load blocks using transformer kVA weights, anchored by RTU readings where available
4. **Solves a MILP** to maximize restored customers while penalizing switching actions and enforcing thermal and radiality constraints
5. **Emits** an ordered switching plan with independent verification
6. **Renders** an ADMS-style one-line network diagram showing switch states and load before and after restoration

## Usage

```bash
python offload_planner.py [season]
```

`season` is optional and defaults to `summer`. Valid values: `winter`, `spring`, `summer`, `fall`.

### Example output

```
=== SUBSTATION OUTAGE OFFLOAD PLAN  (season: summer) ===
Outage: SUB-X fully de-energized

Seasonal feeder maxima (from 1 yr of CB data):
  A1:  160.0 A
  X1:  220.0 A
  ...

--- SWITCHING SEQUENCE ---
Step 1: VERIFY OPEN  CB-X1  (feeder breaker, substation de-energized)
Step 2: OPEN   S-X1-BC  (MANUAL)  sectionalize dead zone between X1-B | X1-C
Step 3: CLOSE  T1  (remote)  backfeed X1-C from SUB-A via feeder A1  (~82 A pickup)
...

--- VERIFICATION ---
  [PASS] outage substation CBs open
  [PASS] final topology is radial (forest)
  [PASS] no component ties two substations together
  [PASS] all sections within thermal rating
```

## Network model

The built-in network (editable at the top of `offload_planner.py`) represents:

| Component | Description |
|---|---|
| `SUB-A`, `SUB-B` | Healthy source substations |
| `SUB-X` | The outaged substation |
| `X1`, `X2` | De-energized feeders to be restored |
| `A1`, `B1`, `B2` | Healthy neighboring feeders available for backfeed |
| Tie switches `T1–T3` | Normally-open backfeed paths |
| Sectionalizing switches | Mid-feeder isolation points |

The MILP objective maximizes restored customers (weighted at 100 per customer) minus switching costs (1 per remote operation, 5 per manual truck roll), subject to:

- Kirchhoff's current law at every node
- Thermal ratings on every edge
- Radiality constraint (switched network must remain a spanning forest)
- No substation paralleling

## Network diagram

Running the script automatically opens an ADMS-style one-line diagram of the post-switching state. You can also call `draw_network()` directly from Python:

```python
from offload_planner import draw_network, solve_offload, allocate_block_loads, ...

# Initial outage state (X feeders dark, ties open)
draw_network()

# Post-switching restored state with load annotations
draw_network(sol=sol, block_amps=block_amps)
```

The diagram uses a dark canvas with feeder color-coding:

| Visual element | Meaning |
|---|---|
| Blue rectangle | Healthy substation bus |
| Red rectangle | Outaged substation (SUB-X) |
| Colored circles | Load blocks, shaded per feeder; label shows block name and customer count |
| Gold annotation | Block load in amps (shown when `block_amps` is provided) |
| Solid feeder-colored line | Energized, closed switch or cable |
| Dark gray line | De-energized section |
| Dashed arc | Tie switch (T1–T3), arced to avoid overlapping feeder lines |
| ● (filled circle on line) | Switch closed |
| × (circled cross on line) | Switch open |

Requires `matplotlib` (optional — the solver and plan output work without it).

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

PuLP ships with the CBC solver, so no external solver installation is required.

## Adapting to a real network

Replace the `NODES`, `EDGES`, and `FEEDERS` dictionaries with your actual network data. RTU-metered switches can be added to `FEEDER_RTUS` to improve load allocation accuracy. Peak targets in `FEEDER_PEAK_TARGET` should come from real SCADA historian exports rather than the synthetic year generator.
