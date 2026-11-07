"""Export derived risk results for the public project website."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SHOCK_FIELDS = ("fx_spot_move", "full_revaluation", "delta_pnl", "delta_gamma_pnl")


def records(path: Path, columns: tuple[str, ...] | None = None) -> list[dict]:
    """Convert a report to JSON records with finite numbers or nulls."""
    frame = pd.read_csv(path, usecols=columns)
    return json.loads(frame.to_json(orient="records", double_precision=10))


def export_site(reports: Path, metadata_path: Path, output: Path) -> None:
    """Package public summaries and the synthetic FX-only shock grid."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    backtest = pd.read_csv(reports / "local" / "backtest.csv", usecols=["date"])
    dates = backtest["date"]
    payload = {
        "market": {
            "as_of": metadata["sample_last_date"],
            "start": metadata["sample_first_date"],
            "observations": metadata["sample_observations"],
            "backtest_start": dates.min(),
            "backtest_end": dates.max(),
        },
        "snapshot": records(reports / "snapshot.csv"),
        "diagnostics": records(reports / "diagnostics.csv"),
        "stress": records(reports / "stress.csv"),
        "curve": records(reports / "local" / "curve.csv", SHOCK_FIELDS),
        "comparison": records(reports / "comparison.csv"),
        "context": json.loads((reports / "context.json").read_text(encoding="utf-8")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--metadata", type=Path, default=Path("data/market_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/results.json"))
    args = parser.parse_args()
    export_site(args.reports, args.metadata, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
