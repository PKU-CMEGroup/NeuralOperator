from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect extracted CPG supersonic-bump mesh files."
    )
    parser.add_argument(
        "case",
        type=Path,
        help="Case directory containing Bump.inp/Bump.msh, or path to Bump.inp",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = args.case.parent if args.case.name.lower().endswith(".inp") else args.case
    inp_path = args.case if args.case.name.lower().endswith(".inp") else case_dir / "Bump.inp"
    msh_path = case_dir / "Bump.msh"

    payload: dict[str, Any] = {
        "kind": "cpg_bump_mesh_inspection",
        "case_dir": str(case_dir),
        "inp": inspect_inp_mesh(inp_path),
    }
    if msh_path.exists():
        payload["msh"] = inspect_gmsh41_counts(msh_path)

    print(f"case: {case_dir}")
    print(f"nodes: {payload['inp']['num_nodes']}")
    print(f"element_blocks: {payload['inp']['element_blocks']}")
    print(f"surface_area_total: {payload['inp']['surface_area_total']:.8g}")
    print(f"surface_nodes_with_area: {payload['inp']['surface_nodes_with_area']}")
    print(f"boundary_lengths: {payload['inp']['boundary_length_by_elset']}")
    if "msh" in payload:
        print(f"gmsh: {payload['msh']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"saved: {args.output}")


def inspect_inp_mesh(path: Path) -> dict[str, Any]:
    nodes: dict[int, np.ndarray] = {}
    elements: list[tuple[int, str, str, list[int]]] = []
    current_section: tuple[str, dict[str, str]] | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            current_section = _parse_section(line)
            continue
        if current_section is None:
            continue
        section, attrs = current_section
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if section == "NODE" and len(parts) >= 4:
            node_id = int(parts[0])
            nodes[node_id] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])])
        elif section == "ELEMENT" and len(parts) >= 3:
            elem_id = int(parts[0])
            elem_type = attrs.get("TYPE", "")
            elset = attrs.get("ELSET", "")
            elements.append((elem_id, elem_type, elset, [int(part) for part in parts[1:]]))

    if not nodes:
        raise ValueError(f"no *NODE records found in {path}")

    element_blocks: dict[str, int] = defaultdict(int)
    boundary_length_by_elset: dict[str, float] = defaultdict(float)
    surface_area_total = 0.0
    node_area = {node_id: 0.0 for node_id in nodes}
    for _, elem_type, elset, conn in elements:
        block_name = f"{elem_type}:{elset}"
        element_blocks[block_name] += 1
        points = np.asarray([nodes[node_id] for node_id in conn], dtype=np.float64)
        if elem_type.upper() == "T3D2" and len(conn) == 2:
            boundary_length_by_elset[elset] += float(np.linalg.norm(points[1] - points[0]))
        elif elem_type.upper() == "CPS4" and len(conn) == 4:
            area = _quad_area(points[:, :2])
            surface_area_total += area
            share = area / 4.0
            for node_id in conn:
                node_area[node_id] += share

    nonzero_area = np.asarray([value for value in node_area.values() if value > 0.0])
    coordinates = np.asarray(list(nodes.values()), dtype=np.float64)
    return {
        "path": str(path),
        "num_nodes": len(nodes),
        "node_id_min": min(nodes),
        "node_id_max": max(nodes),
        "bounds_min": coordinates.min(axis=0).tolist(),
        "bounds_max": coordinates.max(axis=0).tolist(),
        "num_elements": len(elements),
        "element_blocks": dict(sorted(element_blocks.items())),
        "surface_area_total": float(surface_area_total),
        "surface_nodes_with_area": int(nonzero_area.size),
        "surface_node_area_min": float(nonzero_area.min()) if nonzero_area.size else None,
        "surface_node_area_max": float(nonzero_area.max()) if nonzero_area.size else None,
        "surface_node_area_sum": float(nonzero_area.sum()) if nonzero_area.size else 0.0,
        "boundary_length_by_elset": {
            key: float(value) for key, value in sorted(boundary_length_by_elset.items())
        },
    }


def inspect_gmsh41_counts(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    physical_names: dict[str, str] = {}
    nodes_header: list[int] | None = None
    elements_header: list[int] | None = None
    for index, line in enumerate(lines):
        if line == "$PhysicalNames":
            count = int(lines[index + 1])
            for offset in range(count):
                parts = lines[index + 2 + offset].split(maxsplit=2)
                physical_names[f"{parts[0]}:{parts[1]}"] = parts[2].strip().strip('"')
        elif line == "$Nodes":
            nodes_header = [int(value) for value in lines[index + 1].split()]
        elif line == "$Elements":
            elements_header = [int(value) for value in lines[index + 1].split()]
    return {
        "path": str(path),
        "physical_names": physical_names,
        "nodes_header": nodes_header,
        "elements_header": elements_header,
    }


def _parse_section(line: str) -> tuple[str, dict[str, str]]:
    tokens = [token.strip() for token in line.lstrip("*").split(",")]
    section = tokens[0].upper()
    attrs: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            attrs[key.strip().upper()] = value.strip()
    return section, attrs


def _quad_area(points: np.ndarray) -> float:
    return _triangle_area(points[[0, 1, 2]]) + _triangle_area(points[[1, 2, 3]])


def _triangle_area(points: np.ndarray) -> float:
    ab = points[1] - points[0]
    ac = points[2] - points[0]
    return 0.5 * abs(ab[0] * ac[1] - ab[1] * ac[0])


if __name__ == "__main__":
    main()
