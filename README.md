# Overtake
An AI-powered F1 race strategy and performance analytics platform that uses telemetry, tyre degradation prediction, lap-time forecasting, and race simulation to optimize pit-stop strategies and predict race outcomes.

**Current progress, goals and next steps:** see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Setup (once per clone)

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
git config core.hooksPath .githooks              # enforces PROJECT_STATUS.md updates on every commit
python -m src.ingestion.run_ingestion            # download + build race data (~30 s/race first time)
python -m pytest tests
```
