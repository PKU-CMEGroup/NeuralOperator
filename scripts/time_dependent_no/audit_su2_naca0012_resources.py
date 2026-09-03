"""Audit the pinned SU2 Unsteady NACA0012 Stage-0 resource bundle.

Example
-------

.. code-block:: powershell

   python scripts/time_dependent_no/audit_su2_naca0012_resources.py `
     --resource-dir artifacts/time_dependent_no/su2_naca0012_resources
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.su2_restart_contract import (
    audit_unsteady_naca0012_bundle,
)

DEFAULT_MANIFEST = (
    REPO_ROOT / "docs/time_dependent_no/R0_SU2_NACA_RESOURCE_MANIFEST.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit_unsteady_naca0012_bundle(
        resource_dir=args.resource_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
