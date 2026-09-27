"""The console runner end to end, at pinned install dates so the output does
not depend on the day the tests run."""

from run_fixtures import main


def run(capsys, install_date: str) -> str:
    main(["--install-date", install_date])
    return capsys.readouterr().out


def test_the_install_date_sets_the_rebate(capsys):
    before = run(capsys, "2026-12-31")
    after = run(capsys, "2027-01-01")
    assert "installed 2026-12-31" in before
    assert "deeming factor 6.8 applies" in before
    assert "installed 2027-01-01" in after
    assert "deeming factor 5.7 applies" in after
