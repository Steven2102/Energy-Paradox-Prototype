"""The dispatch simulation: the source of kwh_shifted, r_out and r_in.

    kwh_shifted = Σ discharge_t
    r_out       = Σ(discharge_t × rate_t) / Σ discharge_t    what the energy was worth on release
    r_in        = Σ(charge_t × cost_t) / Σ charge_t          what it cost to store

cost_t is the import rate when the battery charged from the grid, and the
feed-in tariff forgone when it charged from solar surplus. Both rates are
averages over the half-hours in which the battery actually moved energy.

How the battery decides. Every half-hour in which the household would import
is a chance to discharge, worth that half-hour's import rate. Every earlier
half-hour within the planning horizon is a chance to charge: from solar
surplus at the feed-in tariff forgone, or from the grid at the import rate.
The battery serves the most valuable half-hours first, each from the
cheapest earlier energy, for as long as the margin on a kWh clears the
battery's marginal wear:

    rate_d − cost_c / efficiency > marginal_throughput_cost

Wear (config/batteries.yaml) is the degradation one more delivered kWh
causes. It decides what is worth moving, but it is not a cash flow, so it
does not enter the bill saving or the headline equation. Surplus is one input
to the comparison, never the trigger, and no tariff shape is a special case:
on a flat tariff without solar, no earlier energy is ever cheaper, so nothing
moves.

Limits: stored energy never exceeds capacity; at most power_kw × 0.5 kWh goes
in or out in a half-hour; the round-trip loss is taken on the way in, so
charging c kWh stores efficiency × c; and the battery never charges and
discharges in the same half-hour. State of charge carries from each
half-hour to the next.
"""

from dataclasses import dataclass

import numpy as np

from src.generator import Year

HOURS_PER_INTERVAL = 0.5
EPSILON = 1e-12  # kWh; anything smaller is treated as zero


@dataclass(frozen=True)
class Dispatch:
    """A year of battery operation. Arrays are kWh per half-hour."""

    discharge: np.ndarray     # delivered to the household
    charge_grid: np.ndarray   # drawn from the grid into the battery
    charge_solar: np.ndarray  # solar surplus stored instead of exported
    stored: np.ndarray        # held at the end of each half-hour
    imports: np.ndarray
    exports: np.ndarray

    # The dispatch terms of the headline equation.
    kwh_shifted: float
    r_out: float | None       # None when the battery moved nothing
    r_in: float | None

    # The energy part of the bill, $/yr: imports × rate − exports × feed-in.
    # Daily and controlled-load charges are left out; a battery changes neither.
    bill_without: float
    bill_with: float


def simulate(year: Year, *, capacity_kwh: float, power_kw: float, efficiency: float,
             marginal_throughput_cost_aud_per_kwh: float, horizon_intervals: int) -> Dispatch:
    wear = marginal_throughput_cost_aud_per_kwh  # $ per delivered kWh
    need = np.maximum(year.load - year.solar, 0.0)     # imported without a battery
    surplus = np.maximum(year.solar - year.load, 0.0)  # exported without a battery
    rate = year.import_rate
    step = power_kw * HOURS_PER_INTERVAL               # kWh in or out per half-hour

    discharge = np.zeros(len(need))
    charge_grid = np.zeros(len(need))
    charge_solar = np.zeros(len(need))
    stored = np.zeros(len(need))

    # Most valuable half-hours first; on a tie, the earlier one.
    for d in sorted(np.flatnonzero(need > EPSILON), key=lambda t: (-rate[t], t)):
        first = max(0, d - horizon_intervals)
        if first == d or charge_grid[d] + charge_solar[d] > 0:
            continue  # nothing earlier to draw on, or already charging here
        earlier = slice(first, d)
        while True:
            wanted = min(need[d], step) - discharge[d]
            if wanted <= EPSILON:
                break
            # How much each earlier half-hour could still supply, in kWh delivered
            # at d: its unused charging power, and the capacity left free at every
            # half-hour from then until d.
            room = np.minimum(
                (step - charge_grid[earlier] - charge_solar[earlier]) * efficiency,
                capacity_kwh - _highest_from_each_point_on(stored[earlier]),
            )
            room[discharge[earlier] > 0] = 0.0
            from_solar = np.minimum(room, (surplus[earlier] - charge_solar[earlier]) * efficiency)
            solar_cost = np.where(from_solar > EPSILON, year.feed_in, np.inf)
            grid_cost = np.where(room > EPSILON, rate[earlier], np.inf)

            s, g = _cheapest(solar_cost), _cheapest(grid_cost)
            if solar_cost[s] <= grid_cost[g]:
                i, cost, available, source = s, solar_cost[s], from_solar[s], charge_solar
            else:
                i, cost, available, source = g, grid_cost[g], room[g], charge_grid
            if rate[d] - cost / efficiency <= wear:
                break  # no earlier energy left that beats importing now by more than wear

            amount = min(wanted, available)
            c = first + i
            source[c] += amount / efficiency
            stored[c:d] += amount
            discharge[d] += amount

    imports = need - discharge + charge_grid
    exports = surplus - charge_solar
    kwh_shifted = float(discharge.sum())
    if kwh_shifted > 0:
        r_out = float((discharge * rate).sum() / kwh_shifted)
        r_in = float(((charge_grid * rate).sum() + (charge_solar * year.feed_in).sum())
                     / (charge_grid + charge_solar).sum())
    else:
        r_out = r_in = None

    return Dispatch(
        discharge=discharge,
        charge_grid=charge_grid,
        charge_solar=charge_solar,
        stored=stored,
        imports=imports,
        exports=exports,
        kwh_shifted=kwh_shifted,
        r_out=r_out,
        r_in=r_in,
        bill_without=_energy_bill(need, surplus, year),
        bill_with=_energy_bill(imports, exports, year),
    )


def _highest_from_each_point_on(values: np.ndarray) -> np.ndarray:
    """[1, 3, 2] -> [3, 3, 2]: the most the battery holds from each half-hour to the end."""
    return np.maximum.accumulate(values[::-1])[::-1]


def _cheapest(costs: np.ndarray) -> int:
    """Index of the lowest cost, taking the latest on a tie: energy held for less
    time ties up less capacity."""
    return len(costs) - 1 - int(np.argmin(costs[::-1]))


def _energy_bill(imports: np.ndarray, exports: np.ndarray, year: Year) -> float:
    return float((imports * year.import_rate).sum() - (exports * year.feed_in).sum())
