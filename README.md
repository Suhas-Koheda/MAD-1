# Evidence-MAD: Evidence-Triggered Multi-Agent Debate with Video Fact-Checking

A research prototype for evidence-triggered, conflict-type-aware multi-agent debate with speaker diarization for automated video fact-checking.

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your Groq API key

# Run everything (frontend + backend)
./run.sh
```

- **Frontend**: http://127.0.0.1:3000
- **Video Mode**: http://127.0.0.1:3000/video.html
- **API Docs**: http://127.0.0.1:8000/docs

## Evaluate

```bash
python run_evaluation.py        # Demo mode (10 tests, 100% pass)
python run_real_evaluation.py   # Live mode with Groq API
python generate_graphs.py       # Generate research paper graphs
```

## Key Features

- **9 Search Agents**: Web, Wikipedia, Crossref, News, GitHub, FactCheck, Reddit, YouTube
- **Evidence-Triggered Debate**: Skips debate when sources agree (saves ~60% LLM calls)
- **5 Conflict Types**: Factual, Temporal, Version, Contextual, Source
- **Video Fact-Checking**: Speaker diarization + transcription + claim extraction
- **RawBlock UI**: Brutalist black-on-white design

## Paper

See [PAPER.md](PAPER.md) for the full research paper and [REPORT.md](REPORT.md) for the project report.
