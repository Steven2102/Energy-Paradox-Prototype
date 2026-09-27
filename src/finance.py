"""The headline equation, and the two equations behind its cost side.

    annual_saving = kwh_shifted × (r_out − r_in / efficiency) + demand_saving
    payback_years = (battery_cost − rebate) / annual_saving

    battery_cost  = fixed + variable × kWh
    rebate        = Σ over taper bands of (kWh in band × share) × deeming_factor × stc_price

Every input is computed or looked up elsewhere: kwh_shifted, r_out and r_in by
the dispatch simulation, demand_saving by the demand-charge model, and the
cost and rebate parameters from config/, including the deeming factor in
effect on the install date. This module only does the arithmetic, and keeps
every intermediate term on its results so a justification can walk through
them with a household's own figures.

This is simple payback, as the equation says: no price growth, discounting or
degradation.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


def battery_cost_aud(usable_kwh: float, fixed_aud: float, variable_aud_per_kwh: float) -> float:
    """Installed price before rebate: fixed + variable × kWh. A flat $/kWh would
    make small batteries look far cheaper than they are."""
    return fixed_aud + variable_aud_per_kwh * usable_kwh


@dataclass(frozen=True)
class Rebate:
    """Every term of the rebate equation."""

    usable_kwh: float
    deeming_factor: float                   # STCs per kWh, in effect on the install date
    stc_price_aud: float                    # $ per STC
    bands: tuple[tuple[float, float], ...]  # per taper band: (kWh of this battery in it, share)
    weighted_kwh: float                     # Σ kWh in band × share
    stcs: float                             # weighted_kwh × deeming_factor
    rebate_aud: float                       # stcs × stc_price_aud


def rebate(
    usable_kwh: float,
    *,
    deeming_factor: float,
    stc_price_aud: float,
    taper: Sequence[tuple[float, float]],
) -> Rebate:
    """The federal battery rebate, paid as STCs:

        rebate = Σ over taper bands of (kWh in band × share) × deeming_factor × stc_price_aud

    taper is (up_to_kwh, share) pairs in ascending order. Capacity from the
    previous band's limit (0 for the first) up to up_to_kwh earns that share of
    the deeming factor, so a battery's first kWh earn more than its last.
    Capacity above the last band earns nothing.
    """
    if usable_kwh < 0:
        raise ValueError(f"usable_kwh cannot be negative, got {usable_kwh}")
    bands = []
    lower = 0.0
    for up_to_kwh, share in taper:
        bands.append((max(0.0, min(usable_kwh, up_to_kwh) - lower), share))
        lower = up_to_kwh
    weighted_kwh = sum(kwh * share for kwh, share in bands)
    stcs = weighted_kwh * deeming_factor
    return Rebate(
        usable_kwh=usable_kwh,
        deeming_factor=deeming_factor,
        stc_price_aud=stc_price_aud,
        bands=tuple(bands),
        weighted_kwh=weighted_kwh,
        stcs=stcs,
        rebate_aud=stcs * stc_price_aud,
    )


@dataclass(frozen=True)
class Payback:
    """Every term of the headline equation: its inputs and each step between."""

    # Inputs
    kwh_shifted: float        # kWh/yr the battery delivered
    r_out: float              # $/kWh the delivered energy was worth
    r_in: float               # $/kWh it cost to store
    efficiency: float         # round trip, 0-1
    demand_saving: float      # $/yr from a lower peak demand
    battery_cost: float       # $ installed, before rebate
    rebate: float             # $

    # Steps
    r_in_after_losses: float  # $/kWh: r_in / efficiency, the cost of storing enough to deliver 1 kWh
    saving_per_kwh: float     # $/kWh: r_out − r_in_after_losses
    energy_saving: float      # $/yr: kwh_shifted × saving_per_kwh
    annual_saving: float      # $/yr: energy_saving + demand_saving
    net_cost: float           # $: battery_cost − rebate
    payback_years: float      # net_cost / annual_saving; math.inf if it never pays for itself


def payback(
    *,
    kwh_shifted: float,
    r_out: float,
    r_in: float,
    efficiency: float,
    demand_saving: float,
    battery_cost: float,
    rebate: float,
) -> Payback:
    """Evaluate the headline equation.

    Every argument is keyword-only and required. demand_saving in particular
    has no default, so a caller must decide it rather than have it silently
    assumed to be zero.
    """
    if not 0 < efficiency <= 1:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    if kwh_shifted < 0:
        raise ValueError(f"kwh_shifted cannot be negative, got {kwh_shifted}")
    if not 0 <= rebate <= battery_cost:
        raise ValueError(f"rebate must be between 0 and the battery cost, got {rebate}")

    r_in_after_losses = r_in / efficiency
    saving_per_kwh = r_out - r_in_after_losses
    energy_saving = kwh_shifted * saving_per_kwh
    annual_saving = energy_saving + demand_saving
    net_cost = battery_cost - rebate
    # A battery that saves nothing, or loses money, never pays for itself.
    payback_years = net_cost / annual_saving if annual_saving > 0 else math.inf

    return Payback(
        kwh_shifted=kwh_shifted,
        r_out=r_out,
        r_in=r_in,
        efficiency=efficiency,
        demand_saving=demand_saving,
        battery_cost=battery_cost,
        rebate=rebate,
        r_in_after_losses=r_in_after_losses,
        saving_per_kwh=saving_per_kwh,
        energy_saving=energy_saving,
        annual_saving=annual_saving,
        net_cost=net_cost,
        payback_years=payback_years,
    )
