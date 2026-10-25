default:
	@just --list

setup:
	python3 -m venv --clear .venv
	.venv/bin/pip install -r requirements.txt

run:
	nix-shell --run ".venv/bin/python main.py"

bench: setup
	hyperfine --warmup 3 --runs 10 ".venv/bin/python main.py" --export-json bench.json
