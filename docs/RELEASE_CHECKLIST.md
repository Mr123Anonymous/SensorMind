# Release Checklist

## Code and Data
- [ ] Run Phase 2 preprocessing: python scripts/run_phase2.py
- [ ] Run Phase 3 training: python scripts/run_phase3.py
- [ ] Run Phase 4 analysis: python scripts/run_phase4.py
- [ ] Run tests: pytest tests/ -v
- [ ] Validate Streamlit App locally

## Documentation
- [ ] README.md up to date
- [ ] docs/ARCHITECTURE.md reviewed
- [ ] docs/DEPLOYMENT.md reviewed
- [ ] docs/API.md reviewed
- [ ] docs/CODE_STRUCTURE.md reviewed
- [ ] LICENSE present

## Deployment
- [ ] Streamlit Cloud app configured with entrypoint app/Home.py
- [ ] .streamlit/config.toml validated
- [ ] .streamlit/secrets.toml populated in deployment environment
- [ ] runtime.txt compatible with target Python version

## CI/CD
- [ ] CI workflow passing
- [ ] Coverage report generated
- [ ] Lint and type-check completed

## Repository Finalization
- [ ] Root branch protection enabled (recommended)
- [ ] README badges added (optional)
- [ ] First public release tag created (v0.1.0)
