"""Config and fixture loading: all three fixtures load against their tariffs,
and anything the loaders cannot represent is refused rather than dropped."""

import textwrap

import pytest

from src.config import load_config
from src.profile import check_matches_tariff, load_fixture, load_fixtures
from src.tariff import parse_tariff, unmodelled_components

CONFIG = load_config()
FIXTURES = {profile.name: profile for profile in load_fixtures()}

# What each fixture's tariff must declare as unpriced. household_c triggers the
# declaration while demand charges are unimplemented (CLAUDE.md, test 6).
EXPECTED_UNMODELLED = {
    "reference_household": [],
    "household_b": [],
    "household_c": ["demand"],
}


@pytest.mark.parametrize("name", EXPECTED_UNMODELLED)
def test_fixture_loads_and_its_tariff_can_price_it(name):
    profile = FIXTURES[name]
    check_matches_tariff(profile, CONFIG.tariffs[profile.tariff_ref])


@pytest.mark.parametrize("name, expected", EXPECTED_UNMODELLED.items())
def test_unpriced_components_are_declared(name, expected):
    tariff = CONFIG.tariffs[FIXTURES[name].tariff_ref]
    assert list(unmodelled_components(tariff)) == expected


def test_sourced_config_is_verified_and_keeps_its_ranges():
    batteries, incentives = CONFIG.batteries, CONFIG.incentives
    assert batteries.verified and incentives.verified
    assert incentives.stc_price_range_aud == (33.0, 40.0)
    assert batteries.round_trip_efficiency_range == (0.85, 0.92)
    assert batteries.cost_model.valid_range_kwh == (5.0, 20.0)
    assert batteries.power_kw_range == (3.5, 11.5)
    assert batteries.marginal_throughput_cost_range == (0.005, 0.05)


def test_a_flat_tariff_is_one_window_covering_the_day():
    windows = CONFIG.tariffs["generic_flat_2026"].energy_windows
    assert [(w.name, w.hours) for w in windows] == [("all_day", ((0, 24),))]


def tariff_entry(**changes):
    """The smallest valid tariffs.yaml entry, with changes applied."""
    entry = {
        "plan": "test",
        "gst_included": True,
        "daily_charge_aud": 1.0,
        "windows": {"all_day": {"hours": [[0, 24]], "rate_aud_per_kwh": 0.30}},
    }
    entry.update(changes)
    return entry


def test_an_unrecognised_key_is_refused_not_dropped():
    with pytest.raises(ValueError, match="unrecognised"):
        parse_tariff("test", tariff_entry(network_access_charge_aud=0.5))


@pytest.mark.parametrize("windows", [
    pytest.param({"day": {"hours": [[0, 12]], "rate_aud_per_kwh": 0.30}}, id="gap"),
    pytest.param({"day": {"hours": [[0, 13]], "rate_aud_per_kwh": 0.30},
                  "night": {"hours": [[12, 24]], "rate_aud_per_kwh": 0.20}}, id="overlap"),
])
def test_windows_must_cover_every_half_hour_exactly_once(windows):
    with pytest.raises(ValueError, match="exactly one"):
        parse_tariff("test", tariff_entry(windows=windows))


def test_schema_only_components_are_recorded_and_declared():
    tariff = parse_tariff("test", tariff_entry(block={"threshold_kwh": 20}, seasonal={"summer": {}}))
    assert list(unmodelled_components(tariff)) == ["block", "seasonal"]


def test_rates_excluding_gst_are_refused():
    with pytest.raises(ValueError, match="GST"):
        parse_tariff("test", tariff_entry(gst_included=False))


def test_a_fixture_that_does_not_reconcile_is_refused(tmp_path):
    # The trap in the reference bills: read only the first of two rate blocks
    # and the period comes up 57% short, raising no error of its own.
    path = tmp_path / "short.yaml"
    path.write_text(textwrap.dedent("""\
        meta:
          source: test
        from_bill:
          tariff_ref: generic_flat_2026
          has_solar: false
          annual_kwh_total: 4800
          annual_kwh_by_window:
            all_day: 2064
    """))
    with pytest.raises(ValueError, match="printed total"):
        load_fixture(path)
