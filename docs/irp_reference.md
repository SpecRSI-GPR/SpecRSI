# IRP reference profile

The `reference` profile implements the confirmed 12-instance inventory-routing specification. Its complete parameters are stored in [the configuration](../configs/science/reference/irp.json); the simulator is [irp_reference.py](../src/specrsi/science/irp_reference.py). Select `scenario_set="reference"` explicitly when using the registry API.

## Instances

Each instance fixes retailer locations, capacities, initial inventory, fleet size, daily mileage limits, and a demand seed. The table summarizes the configuration; coordinates in kilometres are listed in full in the JSON. Capacity and initial stock apply to each retailer. The depot is `(0, 0)`.

| Instance | Retailers | Days | Mean daily demand | Demand CV | Capacity | Initial stock | Vehicles | km / vehicle / day | Demand seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| irp_case_01 | 3 | 7 | 20 | 0.15 | 100 | 40 | 1 | 200 | 1001 |
| irp_case_02 | 4 | 7 | 25 | 0.15 | 125 | 50 | 2 | 250 | 1002 |
| irp_case_03 | 6 | 10 | 22 | 0.20 | 110 | 44 | 2 | 300 | 1003 |
| irp_case_04 | 6 | 14 | 22 | 0.20 | 110 | 44 | 2 | 300 | 1004 |
| irp_case_05 | 7 | 10 | 30 | 0.25 | 150 | 60 | 3 | 350 | 1005 |
| irp_case_06 | 7 | 14 | 30 | 0.25 | 150 | 60 | 3 | 350 | 1006 |
| irp_case_07 | 9 | 14 | 28 | 0.30 | 140 | 56 | 3 | 400 | 1007 |
| irp_case_08 | 11 | 14 | 25 | 0.30 | 125 | 50 | 4 | 450 | 1008 |
| irp_case_09 | 11 | 21 | 35 | 0.35 | 175 | 70 | 4 | 500 | 1009 |
| irp_case_10 | 13 | 14 | 32 | 0.35 | 160 | 64 | 5 | 550 | 1010 |
| irp_case_11 | 15 | 21 | 38 | 0.40 | 190 | 76 | 5 | 600 | 1011 |
| irp_case_12 | 15 | 28 | 40 | 0.40 | 200 | 80 | 6 | 650 | 1012 |

Demand is generated with NumPy `PCG64` in day-major, then retailer-ID order. Each draw is normal with standard deviation `mean × CV`, clipped at zero, then rounded down to an integer. The instance's demand seed fixes the entire sequence independently of search or training seeds. All candidate policies and the reference policy use the same sequence.

## Editable candidate

A candidate contains one integer order-up-to target per retailer:

```json
{"target_stock": [60, 60, 60]}
```

The example applies to `irp_case_01`. Each target must lie between zero and the corresponding store capacity. Daily replenishment is `max(target_stock - available_inventory, 0)`. Targets stay fixed during a run; replenishment quantities respond to daily inventory. Current and future realized demand are not exposed to this decision.

This interface edits inventory targets. Next-fit loading, fixed-fleet assignment, and nearest-neighbor routing remain fixed. Arbitrary daily shipment schedules and custom routes are not candidate fields.

## Simulation and constraints

Every day runs in this order:

1. Increase inventory age by one and discard units whose age reaches seven days.
2. Compute replenishment from the remaining inventory and the candidate targets.
3. Dispatch shipments and receive them the same day at age zero.
4. Realize demand and serve it from the oldest inventory first (FIFO).
5. Record end-of-day inventory and holding, shortage, and transportation costs.

Initial stock has age one. Warehouse supply is unlimited. Expiry reduces inventory; it is not counted again as unmet demand or charged a shortage penalty.

Deliveries are grouped in retailer-ID order using **Next-fit**: when an entire store order does not fit in the current 500-unit trip, start another trip. Trips cycle through the configured fleet, restarting from the first vehicle each day. Each trip starts and ends at the depot, visiting its retailers by the Euclidean **nearest-neighbor heuristic**; equal-distance choices use retailer ID. This is a deterministic route, not an exact TSP optimum. Every trip contributes to the vehicle's cumulative daily mileage.

Evaluation enforces store capacities, 500 units per trip, fixed fleet size, each vehicle's daily mileage limit, FIFO expiry, and cumulative service of at least 95%:

```text
service = total satisfied demand / total demand
```

Demand and inventory conservation are also checked. Violations return an invalid-candidate result.

## Cost and score

```text
TC = 0.5 × sum(end-of-day inventory)
   + 10.0 × total unmet demand
   + 3.0 × total distance in km
J  = max(0, 1 - TC / TC_ref)
```

Costs are in currency units, not LLM tokens. The reference policy replenishes every store to capacity using the same routing rules and demand sequence. `TC_ref` is recomputed from that policy. Its own `J` is therefore zero; a feasible lower-cost candidate receives a positive `J`.

The result's `native_score` is `J`. `specrsi_score` is the registry's internal monotone transformation for its bounded outcome interface. Results also include reference cost, a demand hash, cost components, and daily inventories, shipments, routes, per-vehicle distances, demand, shortages, and expiry.

## Run

After installing the repository dependencies, run these commands from the repository root:

```bash
mkdir -p reports
PYTHONPATH=src python -m specrsi.science.irp_reference --all-cases \
  --output reports/irp_reference.json
```

This evaluates the fill-to-capacity reference policy on all 12 cases. Use `--case-id 0` instead of `--all-cases` for the first case (IDs are zero-based).

To evaluate an edited target vector:

```bash
PYTHONPATH=src python - <<'PYTHON'
from specrsi.science.registry import evaluate

result = evaluate("irp", {"target_stock": [60, 60, 60]},
                  scenario_set="reference", case_id=0)
print({key: result.get(key) for key in
       ("valid", "native_score", "service_level", "total_cost", "reference_cost")})
PYTHON
```

Run the simulation checks with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_irp_reference.py
```
