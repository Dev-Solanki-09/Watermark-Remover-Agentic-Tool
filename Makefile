.PHONY: check-q test-q smoke

check-q:
	@python -m ruff check --quiet .
	@python -m mypy --no-error-summary wmclean

test-q:
	@python -m pytest -q --tb=line --no-header --maxfail=3

smoke:
	@python -m wmclean.cli doctor
	@python -c "from tests.fixture_gen import ALL_FIXTURE_GENERATORS; from wmclean.cli import main; from pathlib import Path; import os; p, _, _ = ALL_FIXTURE_GENERATORS['FX_G01'](Path('fixtures')); code = main(['run', str(p), '-o', 'smoke_out.pdf', '--report', 'smoke_report.json']); assert code == 0, f'Exit {code}'; assert Path('smoke_out.pdf').exists(); assert Path('smoke_report.json').exists(); [os.remove(f) for f in ('smoke_out.pdf', 'smoke_report.json') if os.path.exists(f)]; print('SMOKE TEST PASSED (exit 0)')"
