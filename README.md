# Household Energy Decision Assistant

A decision-support web application for Australian households (South East Queensland)
weighing a home battery purchase. The user uploads an electricity bill and answers a
short questionnaire; the system returns a recommendation justified against that
household's own numbers, and answers follow-up questions about it.

Research prototype, QUT BEAM Group. Showcase: **12 October 2026**.

`CLAUDE.md` is the build specification. Read it before changing anything — it records
what is agreed, what is deliberately out of scope, and several domain facts that cause
plausible-looking wrong code if you do not know them.

## Design in one line

All arithmetic runs deterministically in Python. The language model reads the bill,
classifies follow-up questions and writes prose — it never calculates. Every figure
shown to a user traces back to a visible equation:

```
annual_saving = kwh_shifted × (r_out − r_in / efficiency) + demand_saving
payback_years = (battery_cost − rebate) / annual_saving
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add an API key from stage 6 onwards
```

## Layout

```
config/      market values — tariffs, incentives, battery catalogue, assumptions
fixtures/    reference households the engine is developed and tested against
data/        extracted bill data (aggregate only)
tools/       one-off utilities, e.g. the retailer-specific bill parser
src/         the engine
tests/       see "Required tests" in CLAUDE.md
```

## Fixtures

Three households, which disagree on purpose:

| Fixture | Tariff | Solar | Peak season | Expected |
|---|---|---|---|---|
| `reference_household.yaml` | time-of-use, 4 components | no | winter | battery **not now** |
| `household_b.yaml` | flat, one window | 6.6 kW | summer | battery **worth it** |
| `household_c.yaml` | demand charge | no | summer | **declare** the unmodelled component |

Every test runs against all three. If a change makes the first two agree, the engine has
stopped discriminating and is fitted to one bill.

`reference_household.yaml` is derived from 18 real redacted bills. The other two are
synthetic and labelled as such in their headers.

## Data handling

Source PDFs are **not** in this repository and must not be added — see `.gitignore`.
They contain a real household's consumption data. What is committed is the aggregate
extraction (`data/bills_extracted.csv`) and the fixture, neither of which identifies
anyone.

Place bills locally in `data/bills/` to re-run `tools/parse_globird.py`.

## Build order

See `CLAUDE.md`. Stage 4 is the floor — from there a working, justified recommendation
exists with real numbers.
