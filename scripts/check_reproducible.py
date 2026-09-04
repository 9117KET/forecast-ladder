"""Check that results/published/ is still what results/raw/ produces.

    python scripts/check_reproducible.py

Re-derives every published table from the committed forecasts into a scratch directory and
compares it against what is in the repository. Exits non-zero on a real difference.

**Why this is not `git diff --exit-code`.** It was, and that was wrong. The published tables
are not reproducible byte-for-byte across BLAS and numpy versions: summing 33,600 newvendor
costs in a different order moves the total by one unit in the last place, so
`eur_per_series_day` comes out as 1.1921662362967191 on one machine and ...194 on another.
Every metric is bit-identical, the ranking is identical, and nothing anyone would quote
changes. A byte comparison fails on that, and a check that cries wolf on the sixteenth
significant digit gets switched off the first time it fires.

So numeric columns are compared with a tolerance and everything else exactly. The tolerance
is tight enough that a real change cannot hide under it: 1e-9 relative is still six orders
of magnitude finer than the third decimal place any of these tables are read to.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PUB = ROOT / "results" / "published"

#: Relative tolerance on floating-point columns. Chosen to absorb summation-order noise
#: (which lands around 1e-16 relative) with orders of magnitude to spare, while still
#: catching any difference that could reach a reported figure.
RTOL = 1e-9


def compare(name: str, expected: pd.DataFrame, actual: pd.DataFrame) -> list[str]:
    """Differences between two tables, as human-readable lines. Empty means they agree."""
    problems = []
    if list(expected.columns) != list(actual.columns):
        return [
            f"{name}: columns differ\n"
            f"    committed: {list(expected.columns)}\n"
            f"    rebuilt:   {list(actual.columns)}"
        ]
    if len(expected) != len(actual):
        return [f"{name}: {len(expected)} rows committed, {len(actual)} rebuilt"]

    for col in expected.columns:
        exp, act = expected[col], actual[col]
        if pd.api.types.is_numeric_dtype(exp) and pd.api.types.is_numeric_dtype(act):
            e, a = exp.to_numpy(dtype=float), act.to_numpy(dtype=float)
            both_nan = np.isnan(e) & np.isnan(a)
            close = np.isclose(e, a, rtol=RTOL, atol=0.0, equal_nan=True)
            bad = ~(close | both_nan)
            if bad.any():
                i = int(np.argmax(bad))
                problems.append(
                    f"{name}.{col}: {int(bad.sum())} of {len(e)} values differ by more than "
                    f"{RTOL:g} relative; first at row {i}: {e[i]!r} committed, {a[i]!r} rebuilt"
                )
        else:
            neq = exp.astype(str).to_numpy() != act.astype(str).to_numpy()
            if neq.any():
                i = int(np.argmax(neq))
                problems.append(
                    f"{name}.{col}: {int(neq.sum())} values differ; first at row {i}: "
                    f"{exp.iloc[i]!r} committed, {act.iloc[i]!r} rebuilt"
                )
    return problems


def main() -> int:
    committed = {p.stem: pd.read_csv(p) for p in sorted(PUB.glob("*.csv"))}
    committed_head = json.loads((PUB / "headline.json").read_text(encoding="utf-8"))
    if not committed:
        print(f"nothing to check: no tables in {PUB}")
        return 1

    scratch = Path(tempfile.mkdtemp(prefix="forecast-ladder-repro-"))
    backup = scratch / "committed"
    try:
        # analyse.py writes into results/published/ by design, so the committed copies are
        # moved aside and restored afterwards. The working tree is left exactly as found,
        # including when the rebuild fails.
        shutil.copytree(PUB, backup)
        proc = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts" / "analyse.py")],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:], file=sys.stderr)
            print("\nFAIL: scripts/analyse.py did not complete")
            return proc.returncode

        rebuilt = {p.stem: pd.read_csv(p) for p in sorted(PUB.glob("*.csv"))}
        rebuilt_head = json.loads((PUB / "headline.json").read_text(encoding="utf-8"))

        problems = []
        missing = set(committed) - set(rebuilt)
        if missing:
            problems.append(f"tables no longer produced: {sorted(missing)}")
        extra = set(rebuilt) - set(committed)
        if extra:
            problems.append(f"tables produced but not committed: {sorted(extra)}")
        for name in sorted(set(committed) & set(rebuilt)):
            problems += compare(name, committed[name], rebuilt[name])

        # headline.json carries the figures every write-up quotes, so its string fields are
        # compared exactly and its numbers to the same tolerance as the tables.
        for key in sorted(set(committed_head) | set(rebuilt_head)):
            if key not in committed_head or key not in rebuilt_head:
                problems.append(f"headline.json: key {key!r} present in only one version")
                continue
            c, r = committed_head[key], rebuilt_head[key]
            if isinstance(c, float) and isinstance(r, float):
                if not np.isclose(c, r, rtol=RTOL, atol=0.0, equal_nan=True):
                    problems.append(f"headline.json.{key}: {c!r} committed, {r!r} rebuilt")
            elif c != r:
                problems.append(f"headline.json.{key}: {c!r} committed, {r!r} rebuilt")
    finally:
        for p in PUB.iterdir():
            p.unlink()
        for p in backup.iterdir():
            shutil.copy2(p, PUB / p.name)
        shutil.rmtree(scratch, ignore_errors=True)

    if problems:
        print("FAIL: results/published/ is not what results/raw/ produces\n")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nIf the change is intended, run scripts/analyse.py and commit the result.\n"
            f"Numeric columns are compared at {RTOL:g} relative tolerance, so this is not "
            "floating-point noise."
        )
        return 1

    print(
        f"OK: all {len(committed)} tables and headline.json re-derive from results/raw/ "
        f"within {RTOL:g} relative tolerance"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
