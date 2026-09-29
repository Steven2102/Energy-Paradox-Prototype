"""What more than one test module needs: the fixtures, the install date the tests
pin, and each fixture's recommendation, computed once per test session."""

from datetime import date
from functools import cache
from pathlib import Path

from src.config import load_config
from src.engine import evaluate
from src.profile import load_fixtures
from src.recommend import recommend

CONFIG = load_config()
FIXTURES = {profile.name: profile for profile in load_fixtures()}
NAMES = ["reference_household", "household_b", "household_c"]
INSTALLED = date(2026, 9, 29)
RECORDINGS = Path(__file__).parent / "recordings"  # the model replies the tests replay


def run(profile, tariff=None, config=CONFIG):
    return evaluate(profile, tariff or CONFIG.tariffs[profile.tariff_ref], config, INSTALLED)


@cache
def recommendation(name):
    return recommend(run(FIXTURES[name]))
