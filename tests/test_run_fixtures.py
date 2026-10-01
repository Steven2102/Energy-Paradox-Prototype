"""The console runner end to end, at pinned install dates so the output does
not depend on the day the tests run."""

from run_fixtures import main, read_questions


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


def test_a_questions_file_is_one_question_a_line_without_blanks_or_comments(tmp_path):
    path = tmp_path / "questions.txt"
    path.write_text("# rehearsed for the showcase\nWhy so long?\n\n  What about 13 kWh?  \n")
    assert read_questions(path) == ["Why so long?", "What about 13 kWh?"]
