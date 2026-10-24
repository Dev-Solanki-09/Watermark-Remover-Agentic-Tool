from wmclean.cli import main


def test_cli_dewarp_help(capsys):
    ret = main(["dewarp", "--help"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "dewarp" in out.lower() or "rectify" in out.lower()


def test_cli_run_help_has_dewarp_option(capsys):
    ret = main(["run", "--help"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "--dewarp" in out


def test_cli_doctor_reports_dewarp_extensions(capsys):
    ret = main(["doctor"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "deskew" in out.lower() or "page_dewarp" in out.lower()
