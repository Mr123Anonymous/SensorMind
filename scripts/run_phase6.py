"""Run Phase 6 release-readiness checks.

Checks documentation/deployment artifacts and writes a readiness report.
"""

from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"


def _exists(path: Path) -> bool:
    return path.exists()


def main() -> None:
    checks = {
        "docs_architecture": _exists(DOCS / "ARCHITECTURE.md"),
        "docs_deployment": _exists(DOCS / "DEPLOYMENT.md"),
        "docs_api": _exists(DOCS / "API.md"),
        "docs_code_structure": _exists(DOCS / "CODE_STRUCTURE.md"),
        "license": _exists(ROOT / "LICENSE"),
        "streamlit_config": _exists(ROOT / ".streamlit" / "config.toml"),
        "streamlit_secrets_example": _exists(ROOT / ".streamlit" / "secrets.toml.example"),
        "phase3_summary": _exists(ROOT / "data" / "processed" / "phase3_summary.json"),
        "phase4_error_report_json": _exists(REPORTS / "error_analysis.json"),
        "phase4_error_report_md": _exists(REPORTS / "error_analysis.md"),
        "ci_pipeline": _exists(ROOT / ".github" / "workflows" / "ci-cd.yml"),
    }

    score = sum(1 for value in checks.values() if value)
    total = len(checks)
    ready = score == total

    report = {
        "phase": "phase6",
        "ready": ready,
        "score": score,
        "total": total,
        "checks": checks,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "release_readiness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Phase 6 readiness: {score}/{total}")
    print(f"Ready: {ready}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
