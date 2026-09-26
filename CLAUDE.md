# Household Energy Decision Assistant — project context

## What this is

A **web app** that helps an Australian household (South East Queensland) decide whether a home battery is worth buying. The user uploads their electricity bill and fills in a set of question fields about their household. Those inputs become a **user profile**, which is passed to an LLM via API. The LLM returns a **recommendation that is justified against that specific profile, bill and circumstances**. The household can then **ask follow-up questions** about the recommendation.

Research prototype for a showcase on **12 October 2026**, built by one person in roughly two weeks alongside coursework. Bias every decision toward "working and defensible" over "complete".

Agreed with the project lead on 17 September. Anything not listed here has not been agreed — do not add scope.

## Hard rules

1. **The LLM never calculates.** All arithmetic happens in Python. The model parses the bill, classifies follow-up questions, and writes prose. A hallucinated tariff or payback figure in front of an energy-literate audience is unrecoverable.
2. **The calculation must be expressible as an equation a person can follow.** This is a requirement, not a preference — the author has to be able to explain on a slide how a recommendation was reached. No step may be a black box.
3. **Every number in a justification or answer must be traceable to a computed value.** Prose is generated *from* results, never alongside them.
4. **A question about a changed circumstance triggers a recomputation, never an estimate.** If the model answers "what if I get an EV?" by reasoning rather than re-running the engine, rule 1 has been broken.
5. **Every output is tailored** to this household's bill and profile. Generic advice is a failure, not a fallback.
6. **"Not now" is a valid recommendation** and must be reachable. A system that always recommends buying is advertising.
7. **No autonomous action.** No switching retailers, no ordering anything, no external side effects. This is a sensemaking tool and a chatbot.
8. **Assumptions live in `config/`**, never hardcoded in logic.
9. **A charge on the bill that the engine cannot model must be declared, never ignored.** Parse it, name it, state that this version does not price it, and label the result a partial picture. Silently dropping a third of someone's bill and returning a confident payback is the worst failure this system can produce.
10. **No household-specific constants, anywhere.** If a number was chosen because it made the reference household come out right, it is a bug. Market values go in `config/`; household values come from the bill and the form.

## This must work for households it has never seen

The showcase runs on one real household. The system is not for that household. Every design decision is judged by whether it survives a different bill, a different tariff and a different profile — a household with solar, on a flat rate, peaking in summer, with a demand charge, or all of those at once.

The concrete defence is **three fixtures, and every test runs against all of them**:

| Fixture | Tariff | Solar | Peak season | Expected answer |
|---|---|---|---|---|
| `reference_household.yaml` | time-of-use, 4 components | no | winter | battery **not now** |
| `household_b.yaml` | flat, one window | 6.6 kW | summer | battery **worth it** |
| `household_c.yaml` | demand charge | no | summer | **declare** the unmodelled component |

They disagree on purpose. A recommendation engine tuned to one household returns "not now" for `household_b` as well, and that is immediately visible. If all three cannot pass, the code is fitted to one bill.

`household_b` and `household_c` are synthetic and labelled as such. Only `reference_household.yaml` comes from real bills.

## The calculation, and how it stays explicable

The headline equation, which must be inspectable end to end:

```
annual_saving  = kwh_shifted × (r_out − r_in / efficiency) + demand_saving
payback_years  = (battery_cost − rebate) / annual_saving
```

The two rates are **outputs of the simulation, not inputs** — each is a weighted average over the intervals where the battery actually moved energy:

```
r_out = Σ(discharge_t × rate_t) / Σ(discharge_t)       what the energy was worth on release
r_in  = Σ(charge_t   × cost_t) / Σ(charge_t)           what it cost to store
```

`cost_t` is the import rate when the battery charged from the grid, and the **feed-in tariff forgone** when it charged from solar surplus. A household with both gets a blend, computed rather than assumed. This is what makes the same one-line equation hold for a flat tariff, a time-of-use tariff, a solar household and a no-solar household without special cases: the tariff shape changes the weights, never the form.

`demand_saving` is an additive term for tariffs with a $/kW charge — the reduction in the highest half-hourly demand, summed over billing periods. Zero for tariffs without one. It is not expressible as a $/kWh rate and cannot be folded into the terms above.

Everything else exists to produce `kwh_shifted`, `r_out` and `r_in` honestly. A half-hourly dispatch simulation computes them, but the simulation is **the source of terms in a visible equation**, not a black box that emits an answer. The justification should be able to walk through this arithmetic with the household's real figures.

The simulation is **decided, not optional** — it is what makes the shifted-kWh figure defensible, and every counterfactual depends on being able to re-run it.

*Fallback of last resort, only if dispatch is still not working by day 9: estimate `kwh_shifted` from the bill's window totals. Same equation, much weaker — and biased. Throughput is a `min()` of available cheap energy, expensive-window demand and battery capacity, and a minimum of averages always exceeds the average of minimums, so the shortcut systematically **overstates** what a small battery captures and **understates** the value of a larger one. This is observable in the reference household: averaging assigns identical value to every size above 6 kWh, which is an artefact of the method, not a fact. Ask before switching.*

## Tariffs: a list of components, not a shape

**A tariff is a list of charge components.** A flat tariff is a window list of length one — not a special case, and not a separate code path. Building it the other way round produces a dispatch loop that cannot handle the real bills we have.

Every component type below must exist **in the data model from the start**, even where it is not implemented. A charge type with no slot cannot be detected, and an undetected charge type is silently dropped from someone's bill — which hard rule 9 forbids.

| Component | Status for October |
|---|---|
| `energy_windows` — $/kWh by time of day | implement |
| `daily_charge` — $/day | implement (unaffected by a battery) |
| `controlled_load` — separate circuit, $/kWh | implement; **exclude from addressable consumption** |
| `feed_in` — $/kWh exported | implement as a scalar; time-varying export is schema only |
| `demand` — $/kW on peak half-hourly demand | schema + **declare**; implement if time allows |
| `block` — $/kWh after N kWh | schema + **declare** |
| `seasonal` — rates differing by month | schema + **declare** |

Two consequences for dispatch:

- **A battery has two possible charging sources**, not one: solar surplus *and* cheap grid import. An implementation that only charges from surplus cannot evaluate `reference_household.yaml` at all; one that only charges from cheap grid import cannot evaluate `household_b.yaml`.
- **Dispatch decides by comparing the cost of energy across intervals**, not by asking "is there surplus". Surplus is one input to that comparison, not the trigger.

Demand charges deserve particular care, because they invert the usual result: shaving 3 kW off one half-hour a month can be worth more than a year of energy arbitrage. A system that models only energy will under-recommend batteries for those households — confidently, and in the wrong direction.

## The data pipeline

| Source | Provides |
|---|---|
| **Bill upload** (PDF) | Billing period, total kWh, **consumption split by tariff window**, rates per window, daily charge, controlled load, feed-in tariff, retailer, whether solar exists |
| **Question fields** | Household size, occupancy, appliances, system details, plans (an EV?), priorities, expected years in the home |
| **Generator** | Distribution of consumption *within* each window |

On a **time-of-use** bill, the upload gives magnitude *and* a coarse shape — window totals per billing period, across as many periods as the user supplies. The generator then only distributes within known buckets, anchored to real totals.

On a **flat-tariff** bill there is no window split at all: magnitude only. The generator has to lean entirely on the form, and the result carries materially more uncertainty. **The system must say so** rather than presenting both cases with equal confidence. This asymmetry is the single largest source of error variation between users, and `household_b.yaml` exists partly to keep it visible.

**Suggested, not required:** a confirmation step between extraction and computation — show what was read from the bill, let the user correct it, then compute. Cheap, and the main defence against a silent extraction error producing a confident wrong answer.

## Reference data

`fixtures/reference_household.yaml` is built from 18 real redacted bills (Apr 2025 – Aug 2026, 28-day cycles). It separates what the bill supplies from what the form must still ask, and records the known traps. `household_b.yaml` and `household_c.yaml` are synthetic contrast cases — see the generalisation section above.

The reference bills are **text PDFs** — `pdftotext -layout` reads them reliably, and a deterministic parser for that retailer is in the repo. Treat it as a **fast path, not the design**: extraction must work on retailers and layouts nobody has seen, which is the LLM's job. Structure `bill.py` as a registry of format-specific extractors with the LLM as the general fallback, never as one parser with the LLM bolted on.

## Domain facts that shape design

Facts about the Queensland market and the reference data, not scope decisions.

- **Do not assume a flat tariff.** The reference household is on a four-component time-of-use plan: peak 16:00–21:00 at 43.78 c, offpeak 09:00–16:00 at 25.85 c, shoulder 21:00–09:00 at 28.05 c, controlled load at 20.46 c. The cheap window is **midday, not overnight** — a solar-sponge design.
- **A battery can be worthwhile without solar** where a peak/offpeak spread exists. Here it is 17.93 c/kWh, and the windows are adjacent: charge 09:00–16:00, discharge 16:00–21:00.
- **On a genuinely flat tariff with no solar, a battery is worth close to nothing.** That path must still exist and must be reachable.
- **Battery value is often demand-limited, not capacity-limited.** The reference household consumes 4.3 kWh/day in the peak window, so capacity beyond roughly 6 kWh has little left to shift. Commercial calculators routinely propose 10–13 kWh to households like this. Surfacing that gap is close to the point of the artifact.
- **Do not hardcode a peak season.** The reference household is **winter**-peaking — 34.3 kWh/day in August against 14.8 in March, a 2.3× swing driven by overnight shoulder load. Derive seasonality from the uploaded bills; never from an assumed Brisbane profile.
- **One year of bills is not a stable baseline.** Same-season year-on-year movement in the reference data reaches ±30%. Do not present a single-year projection as precise.
- **An EV does not on its own make a battery worthwhile.** EV charging lands overnight in the shoulder window, not in the peak window a battery serves, so it does not increase addressable load. The chain that works is: EV → consumption rises → **solar** becomes worthwhile → solar creates the surplus that makes a battery worthwhile. A demo built on "add an EV, battery becomes good" will not survive contact with the arithmetic.
- **Observed price growth is not CPI.** Between 2025 and 2026 this plan's peak rate rose 28% and shoulder 33%. A CPI assumption understates what households have seen; make the growth rate an explicit, visible input.
- SEQ feed-in tariffs are roughly 3–10 c/kWh.
- Battery cost behaves as **fixed + variable × kWh**, not pure $/kWh — a flat per-kWh model misprices small systems badly.
- Rates and rebate values in `config/` are **time-sensitive**; check date stamps before trusting them.

## Repository layout

```
CLAUDE.md
config/
  tariffs.yaml        # window lists and rates; flat = one window
  incentives.yaml     # federal rebate schedule and step-downs
  batteries.yaml      # product catalogue: usable kWh, power, efficiency, price
  assumptions.yaml    # anything estimated rather than measured
fixtures/
  reference_household.yaml          # reference household, from 18 real bills
  household_b.yaml    # synthetic: flat tariff, solar, summer-peaking
  household_c.yaml    # synthetic: demand tariff -- declaration test
data/
  bills/              # sample bills, redacted
  brisbane_tmy.csv    # irradiance, if solar is modelled
src/
  bill.py             # PDF bill -> structured values
  profile.py          # form answers -> user profile
  generator.py        # window totals + form answers -> load shape
  dispatch.py         # -> kwh_shifted
  finance.py          # the headline equation
  recommend.py        # recommendation, including "not now"
  explain.py          # LLM justification, from computed values only
  followup.py         # intent classification -> explain or re-run
tests/
  test_dispatch.py
  test_bill.py
app.py
docs/
  background.md       # earlier brainstorming. NOT specification. Do not build from it.
```

## Required tests

Written before the dispatch loop, and **run against all three fixtures**:

1. **Energy balance closes**: in = out + stored + losses, every interval.
2. **A zero-capacity battery exactly reproduces the no-battery bill.**
3. **A lossless, infinite battery eliminates imports in any period with cheaper energy available earlier.**

For bill extraction:

4. **Window quantities sum to the printed total usage.** A real bill in the sample contains **two rate blocks in one period** (a price change mid-period); reading only the first block understates that period by 57% and raises no error. Reconciliation is what catches it.

Against overfitting — these are what stop the engine learning one household:

5. **The three fixtures produce materially different recommendations.** If `reference_household.yaml` and `household_b.yaml` both return "not now", the engine is not discriminating.
6. **`household_c.yaml` triggers the unmodelled-component declaration** while demand charges are unimplemented, and stops triggering it when they are.
7. **Monotonicity.** Each of these must move the result in the stated direction, on every fixture: more consumption in the expensive window → more shifted; a wider rate spread → shorter payback; a longer expected stay → more favourable; a lower feed-in tariff → a battery looks better where solar exists. These are cheap to write and they catch sign errors and inverted comparisons that the balance test cannot see.

## The follow-up chat

A free-text field **after** the recommendation. This is the part that justifies using a language model at all: a one-shot number with no way to interrogate it does not need one.

Free text on the surface, a closed set of intents underneath. The LLM classifies; Python computes.

| Intent | Behaviour |
|---|---|
| **Explain** — "why 15 years?", "what did you assume about my evenings?" | Answer from the existing results object. No recomputation. |
| **Counterfactual** — "what if I get an EV?", "what about 13 kWh?" | Change the named parameter, **re-run the engine**, answer from the new results. Both the old and the new figure must appear in the answer. |
| **Out of scope** — retailers, brands, anything not modelled | Decline, and say why it cannot be answered. |

Counterfactual parameters are a **closed set defined in code** — add an EV, add solar, change battery size, change tariff type, change electricity price growth. A question that maps to nothing in that set is out of scope.

**Fail closed:** an unrecognised question is out of scope. The default must be to decline, not to improvise.

## Out of scope — do not build

Considered during planning and deliberately excluded. If you find these described in `docs/background.md`, that file is history, not specification.

- Tiered decision models, autonomy calibration, "agentic" behaviour of any kind
- Operational load scheduling (shifting hot water, pool pumps) and action logs
- VPP participation
- Retailer or tariff switching recommendations
- Real interval (NEM12) data ingestion
- Neighbourhood comparison or feeder-level views
- **Conversational intake** — the profile comes from **form fields**, not a chat flow. (This concerns *intake* only. The follow-up chat *after* the recommendation is in scope — see above.)
- Open-ended chat — follow-ups resolve to the closed intent set or are declined
- Break-even presented as a distribution, multi-objective size sweeps, priority weighting
- Model training or fine-tuning of any kind

## Build order

1. Engine skeleton — all three fixtures, the equation, console output. No UI, no LLM.
2. Dispatch — producing `kwh_shifted`, `r_out` and `r_in`, with tests 1–3 and 7.
3. Generator — distribute within window totals; handle the flat-tariff no-shape case.
4. Recommendation, including the "not now" path and the declaration path. Tests 5–6.
5. Bill parsing — registry of extractors, LLM fallback. Test 4 applies.
6. LLM justification from computed values.
7. Web app: upload, question fields, results.
8. Follow-up chat — intent classification and the counterfactual re-run path.
9. Demo scenario and rehearsal.

**Stage 4 is the floor** — from there a working, justified recommendation exists with real numbers from a real household.

Bill parsing is deliberately *after* the engine: it is the only stage with a manual fallback, because `fixtures/reference_household.yaml` already holds hand-verified values. If extraction is unfinished on the day, the demo still runs. Dispatch has no such escape hatch, so it goes first.

## Conventions

- Python 3.11+, pandas for time series. **Streamlit is a suggestion** for the web app, not a requirement — pick whatever gets a working UI fastest.
- 30-minute intervals. One year = 17,520 rows. Do not go finer; it gains nothing here.
- Simulate one representative year and project finances forward rather than simulating ten years of dispatch.
- The engine must be callable with a modified parameter and return a fresh result set — the counterfactual path depends on it. Keep it a pure function of (profile, config), with no hidden state between runs.
- Every module in `src/` must be importable and testable without the LLM.
- Prefer boring, readable code over clever code. This will be read by an examiner, and by the author in three weeks.
