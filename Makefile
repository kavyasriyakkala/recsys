.PHONY: setup pipeline smoke test serve clean

setup:
	pip install -e ".[dev]"

pipeline:
	python scripts/run_pipeline.py

smoke:
	python scripts/run_pipeline.py --smoke

test:
	pytest tests/ -v -k "not slow"

test-all:
	pytest tests/ -v

serve:
	uvicorn src.ts.serve.app:app --port 8000

clean:
	rm -rf data/ml-1m data/ml-1m.zip reports/*.json reports/serve_state.pkl
