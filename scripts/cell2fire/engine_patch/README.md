# Cell2Fire engine modifications (WFEDS)

The Cell2Fire engine source is NOT copied into this repository. The repo only
records the exact original version to use (below) plus these two **optional**
WFEDS modifications, maintained as sequential patches on top of a clean clone;
the Docker build downloads, patches and compiles the engine automatically.
Every in-code change is tagged with the comment marker **`WFEDS`**.

- **Base engine:** `github.com/cell2fire/Cell2Fire` @ commit **`b860bcc`** (clean clone).
- **Patch 1:** `cell2fire_initialburned.patch` - seed an observed burned front at t=0.
- **Patch 2:** `cell2fire_intensity.patch` - dump per-period fireline-intensity grids
  (`--out-intensity`). A **diagnostic only** - it never changes fire spread.

> **History:** patch 2 briefly also carried a `--SuppressionFactors` ROS-damping
> "brake". That physical suppression model was **dropped 2026-07-03** - the fire is
> modelled as free-burning (worst credible case) and suppression/evacuation *measures*
> are handled as decisions in the Phase-5 LLM layer, not as engine physics. Only the
> intensity output (which is calibration-free and feeds the LLM's "where is intervention
> feasible" reasoning) was kept. See `LLM-WFEDS/Decisions/Decision log.md`.

## Apply + rebuild (order matters)
```bash
cd <Cell2Fire repo>            # base commit b860bcc
git apply /path/to/cell2fire_initialburned.patch
git apply /path/to/cell2fire_intensity.patch
cd cell2fire/Cell2FireC
rm -f *.o *.gch && make       # FULL rebuild - see the gotcha below
```
**GOTCHA (cost us a debugging round):** patch 1 adds a field to the `arguments` struct
and patch 2 changes `manageFire`'s signature, both read by several `.o` files. The repo
Makefile's dependency tracking is incomplete, so `make` alone recompiles only the changed
files and leaves an **ABI mismatch** (other objects read the struct at the old offsets →
garbage wind/ROS → negative ROS → the fire won't spread). **Always `rm -f *.o *.gch`
(full rebuild)** - the stale `CellsFBP.h.gch` precompiled header would likewise shadow
header changes. (`make clean` also fails because its target lists a non-existent
`Forest.o`.)

---

# Patch 1 - `--InitialBurned` (observed front seeding)

## Why
Operationally we rarely detect a wildfire at its exact ignition - we usually observe it
**already spread** (e.g. from satellite/VIIRS). A decision-support tool should therefore
be able to start from the **current observed perimeter** and forecast forward N hours,
not only from a point. It also enables a fair **validation**: seed the real fire front at
the end of day N and check whether the model reproduces day N+1.

**Not hardcoded / LLM-selectable:** the feature is a plain optional flag. The analyst (or
the Phase-5 LLM) chooses per run:
- **one point** - fire caught early → `IgnitionPoints.csv` (unchanged default), or
- **a front** - fire caught mid-event → `--InitialBurned <file>`.

The **forecast horizon** ("hours since detection") is the normal run duration
(hourly weather rows / grids), independent of this flag.

## What the flag does
`--InitialBurned <file>`: `<file>` lists **1-based, row-major cell ids**, one per line
(a header line like `Ncell` is ignored). At period 0, each listed cell that is burnable
and not already burnt is **ignited like a normal ignition** (so it spreads from t=0).
Interior cells (surrounded by burnt cells) self-extinguish; only the edge advances.
Default (flag absent) = unchanged point-ignition behaviour.

## Files changed (6)
| File | Change |
|---|---|
| `Cell2FireC/ReadArgs.h` | add `InitialBurnedFile` to the `arguments` struct |
| `Cell2FireC/ReadArgs.cpp` | parse `--InitialBurned`; assign into `args` |
| `Cell2FireC/Cell2Fire.h` | add member `std::vector<int> InitialBurnedCells` |
| `Cell2FireC/Cell2Fire.cpp` | `#include <fstream>`; read the file in the ctor; seed the front in `RunIgnition` |
| `Cell2FireC_class.py` | forward `--InitialBurned` to the C++ binary |
| `utils/ParseInputs.py` | add the `--InitialBurned` CLI argument |

## Verified (North Evia 12 h instance, FPL=1, 6 Bft)
- No flag → **3,199** cells burnt (normal point ignition).
- `--InitialBurned` a 441-cell block (379 burnable) → seeded at period 0, spreads to
  **9,143** cells over 12 h, no crash, 13 hourly grids. Starting from a front → a larger
  fire, as expected.

---

# Patch 2 - fireline-intensity output (`--out-intensity`)

## Why
Base Cell2Fire emits no intensity raster, but its FBP module computes Byram fireline
intensity internally all along. This patch simply **exports** it, per period, as a grid.
It is a **diagnostic** - "how intense / how fightable is the fire here" - that never
changes fire behaviour. The Phase-5 LLM layer uses it to reason about response *measures*
(where holding a line is feasible); the fire itself stays free-burning (worst case).

(The suppressability **classes** - `<350` direct attack · `350–1750` mechanical/aerial ·
`1750–3500` serious control problems · `>3500` indirect only, kW/m - are applied
**downstream in Python**, `cell2fire_adapter.SUPP_THRESHOLDS`, not in the engine.)

## What the flag does
- `--out-intensity` (default off; **without it the engine is bit-identical to unpatched**):
  dumps `Intensity<NN>.csv` grids into the same `Grids/` folder as the `ForestGrid<NN>.csv`
  dumps, same numbering → period-aligned. Cell value = **max FREE Byram head fireline
  intensity (kW/m)** seen while the cell was actively burning up to that period; `0.0` =
  never burned / never active. Byram `I = 300·fc·ros` from the engine's own FBP module
  (`FBPfunc5_NoDebug.c: fire_intensity`); `.fi` is read, never modified.

## Files changed (7)
| File | Change |
|---|---|
| `Cell2FireC/ReadArgs.h` | `OutIntensity` in `arguments` |
| `Cell2FireC/ReadArgs.cpp` | parse `--out-intensity`; `printArgs` |
| `Cell2FireC/CellsFBP.h` | `FIgrid` param on both manage-fire signatures |
| `Cell2FireC/CellsFBP.cpp` | record max free `headstruct.fi` into `FIgrid` in `manageFire` AND `manageFireBBO` (no ROS change) |
| `Cell2FireC/Cell2Fire.h` | member `std::vector<double> maxFI` |
| `Cell2FireC/Cell2Fire.cpp` | `maxFI.assign` in `reset()`; pass `&maxFI` at both call sites; `Intensity<NN>.csv` dump in `outputGrid()` |
| `Cell2FireC/WriteCSV.h/.cpp` | `printCSVDoubleGrid()` - plain row-major `%.1f` grid writer |
| `Cell2FireC_class.py` + `utils/ParseInputs.py` | forward / add `--out-intensity` |

## Note on the dropped suppression brake
An earlier version of this patch also had `--SuppressionFactors f1,f2,f3,f4` (per-class
ROS damping). An experiment (aggressive f=0.15/0.25/0.4/0.5) cut the fire only 203.6 →
167.5 km² and did not close the ~5x over-spread; combined with the decision to treat
suppression as a Phase-5 decision concern (not engine physics), the damping was removed.
See `LLM-WFEDS/Decisions/Decision log.md` 2026-07-03.

## Reverting
- Day-to-day: just omit `--out-intensity` - the engine is then bit-identical to unpatched.
- Source: the WSL clone has the InitialBurned state temp-committed
  (`WFEDS: initialburned baseline`); `git checkout -- .` drops the intensity edits.
