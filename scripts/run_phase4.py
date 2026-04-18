"""Run Phase 4: error analysis and release report generation."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.error_analysis import generate_phase4_reports
from src.utils import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    logger.info("Starting Phase 4 pipeline...")
    try:
        report = generate_phase4_reports()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    logger.info("Phase 4 complete. Generated report keys: %s", list(report.keys()))


if __name__ == "__main__":
    main()
