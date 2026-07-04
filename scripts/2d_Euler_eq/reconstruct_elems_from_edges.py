#!/usr/bin/env python3
"""
Rename edge-like elements.npy files and reconstruct mixed 2D elems.npy files.

Expected directory layout:

  root_dir/
    0/
      points.npy or point.npy or grid.npy
      elements.npy  # actually edge elements, usually rows like [1, i, j]
      features.npy
    1/
      ...

For each numeric subfolder, this script:

  1. reads the old elements.npy edge data,
  2. reconstructs 2D cells from the edge graph and point coordinates,
  3. renames elements.npy to egdes.npy by default,
  4. writes elems.npy with rows like [2, i, j, k, -1] or [2, i, j, k, l].

The face reconstruction assumes the edges form a planar 2D mesh.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


POINT_NAME_CANDIDATES = ("points.npy", "point.npy", "grid.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct mixed 2D elems.npy files from edge-like elements.npy folders."
    )
    parser.add_argument("root_dir", help="Directory containing subfolders named 0, 1, ..., 299.")
    parser.add_argument("--start", type=int, default=0, help="First numeric folder index.")
    parser.add_argument("--count", type=int, default=300, help="Number of numeric folders to process.")
    parser.add_argument(
        "--points-name",
        default=None,
        help="Point coordinate filename. If omitted, tries points.npy, point.npy, then grid.npy.",
    )
    parser.add_argument(
        "--elements-name",
        default="elements.npy",
        help="Existing edge-like file to read and rename.",
    )
    parser.add_argument(
        "--edges-name",
        default="egdes.npy",
        help="Filename for the renamed edge file. Default keeps the requested spelling: egdes.npy.",
    )
    parser.add_argument(
        "--elems-name",
        default="elems.npy",
        help="Filename for reconstructed 2D elements.",
    )
    parser.add_argument(
        "--padding-value",
        type=int,
        default=-1,
        help="Padding value to ignore in connectivity arrays.",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "face", "cycles"),
        default="auto",
        help=(
            "Reconstruction method. auto tries planar face walking first, "
            "then closed 3-edge triangle cycles."
        ),
    )
    parser.add_argument(
        "--min-face-nodes",
        type=int,
        default=3,
        help="Minimum number of nodes per reconstructed face.",
    )
    parser.add_argument(
        "--max-face-nodes",
        type=int,
        default=4,
        help="Maximum number of nodes per reconstructed face. Use 4 for triangle/quad meshes.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip folders/files that are missing instead of failing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing edge/elem output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without renaming or writing files.",
    )
    return parser.parse_args()


def find_point_file(folder: Path, points_name: Optional[str]) -> Path:
    if points_name is not None:
        path = folder / points_name
        if path.exists():
            return path
        raise FileNotFoundError(f"Missing point file: {path}")

    for name in POINT_NAME_CANDIDATES:
        path = folder / name
        if path.exists():
            return path

    candidates = ", ".join(POINT_NAME_CANDIDATES)
    raise FileNotFoundError(f"Missing point file in {folder}. Tried: {candidates}")


def edge_endpoints(edge_like_elements: np.ndarray, *, padding_value: int) -> np.ndarray:
    arr = np.asarray(edge_like_elements, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D edge array, got shape {arr.shape}.")

    if arr.shape[1] == 2:
        edges = arr
    elif arr.shape[1] == 3:
        first_column = arr[:, 0]
        if np.all(first_column == 1):
            edges = arr[:, 1:3]
        else:
            raise ValueError(
                "Expected edge-like elements.npy rows shaped [1, i, j]. "
                f"Got first-column values like {np.unique(first_column[:20])}."
            )
    elif arr.shape[0] == 2:
        edges = arr.T
    elif arr.shape[0] == 3 and np.all(arr[0] == 1):
        edges = arr[1:3].T
    else:
        raise ValueError(
            "Expected edges shaped [E, 2], [2, E], [E, 3], or [3, E]. "
            f"Got shape {arr.shape}."
        )

    valid = np.all(edges != padding_value, axis=1)
    return edges[valid].astype(np.int32, copy=False)


def undirected_unique_edges(edges: np.ndarray, n_nodes: int) -> tuple[np.ndarray, dict[str, int]]:
    seen = set()
    unique_edges = []
    self_loops = 0
    duplicate_or_reverse = 0

    for raw_i, raw_j in np.asarray(edges, dtype=np.int32):
        i = int(raw_i)
        j = int(raw_j)
        if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
            raise ValueError(f"Edge ({i}, {j}) references a node outside [0, {n_nodes}).")
        if i == j:
            self_loops += 1
            continue

        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key in seen:
            duplicate_or_reverse += 1
            continue

        seen.add(key)
        unique_edges.append([a, b])

    stats = {
        "raw_edges": int(np.asarray(edges).shape[0]),
        "undirected_edges": len(unique_edges),
        "self_loops": self_loops,
        "duplicate_or_reverse": duplicate_or_reverse,
    }
    return np.asarray(unique_edges, dtype=np.int32), stats


def build_adjacency(edges: np.ndarray, n_nodes: int) -> list[set[int]]:
    adjacency = [set() for _ in range(n_nodes)]
    for raw_i, raw_j in edges:
        i = int(raw_i)
        j = int(raw_j)
        if i == j:
            continue
        if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
            raise ValueError(f"Edge ({i}, {j}) references a node outside [0, {n_nodes}).")
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def signed_area2(nodes: Sequence[int], xy: np.ndarray) -> float:
    a, b, c = xy[list(nodes)]
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def polygon_signed_area2(nodes: Sequence[int], xy: np.ndarray) -> float:
    pts = xy[list(nodes)]
    x = pts[:, 0]
    y = pts[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def orient_triangle_ccw(nodes: Sequence[int], xy: np.ndarray) -> list[int]:
    center = xy[list(nodes)].mean(axis=0)
    ordered = sorted(
        (int(node) for node in nodes),
        key=lambda node: np.arctan2(xy[node, 1] - center[1], xy[node, 0] - center[0]),
    )
    if signed_area2(ordered, xy) < 0:
        ordered = [ordered[0], ordered[2], ordered[1]]
    return ordered


def reconstruct_faces_by_face_walk(
    edges: np.ndarray,
    points: np.ndarray,
    *,
    min_face_nodes: int,
    max_face_nodes: int,
) -> list[list[int]]:
    edges = np.asarray(edges, dtype=np.int32)
    points = np.asarray(points)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected edge endpoints with shape [E, 2], got {edges.shape}.")
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected point coordinates with shape [N, dim>=2], got {points.shape}.")

    xy = np.asarray(points[:, :2], dtype=np.float64)
    adjacency = build_adjacency(edges, points.shape[0])
    sorted_neighbors = []
    neighbor_positions = []
    for node, neighbors in enumerate(adjacency):
        ordered = sorted(
            neighbors,
            key=lambda nbr: np.arctan2(
                xy[nbr, 1] - xy[node, 1],
                xy[nbr, 0] - xy[node, 0],
            ),
        )
        sorted_neighbors.append(ordered)
        neighbor_positions.append({nbr: index for index, nbr in enumerate(ordered)})

    directed_edge_count = sum(len(neighbors) for neighbors in sorted_neighbors)
    visited = set()
    faces = []
    seen = set()
    area_tol = 1e-14

    for start_u, neighbors in enumerate(sorted_neighbors):
        for start_v in neighbors:
            if (start_u, start_v) in visited:
                continue

            face = []
            prev = start_u
            curr = start_v
            closed = False

            for _ in range(directed_edge_count + 1):
                if (prev, curr) in visited:
                    break
                visited.add((prev, curr))
                face.append(prev)

                curr_neighbors = sorted_neighbors[curr]
                if not curr_neighbors:
                    break
                prev_index = neighbor_positions[curr][prev]
                next_node = curr_neighbors[(prev_index - 1) % len(curr_neighbors)]

                prev, curr = curr, next_node
                if prev == start_u and curr == start_v:
                    closed = True
                    break

            if not closed:
                continue
            if len(face) < min_face_nodes or len(face) > max_face_nodes:
                continue
            if len(set(face)) != len(face):
                continue

            if polygon_signed_area2(face, xy) <= area_tol:
                continue

            key = tuple(sorted(int(node) for node in face))
            if key in seen:
                continue

            seen.add(key)
            faces.append([int(node) for node in face])

    faces.sort(key=lambda face: tuple(sorted(face)))
    return faces


def reconstruct_triangles_by_cycles(edges: np.ndarray, points: np.ndarray) -> list[list[int]]:
    edges = np.asarray(edges, dtype=np.int32)
    points = np.asarray(points)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected edge endpoints with shape [E, 2], got {edges.shape}.")
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected point coordinates with shape [N, dim>=2], got {points.shape}.")

    xy = np.asarray(points[:, :2], dtype=np.float64)
    adjacency = build_adjacency(edges, points.shape[0])
    triangles = []
    area_tol = 1e-14

    for i, neighbors_i in enumerate(adjacency):
        for j in neighbors_i:
            if j <= i:
                continue
            common_neighbors = neighbors_i.intersection(adjacency[j])
            for k in common_neighbors:
                if k <= j:
                    continue
                nodes = orient_triangle_ccw((i, j, k), xy)
                if signed_area2(nodes, xy) <= area_tol:
                    continue
                triangles.append(nodes)

    triangles.sort(key=lambda tri: tuple(sorted(tri)))
    return triangles


def reconstruct_cells_from_edges(
    edges: np.ndarray,
    points: np.ndarray,
    *,
    method: str = "auto",
    min_face_nodes: int = 3,
    max_face_nodes: int = 4,
) -> tuple[list[list[int]], str]:
    if method == "face":
        cells = reconstruct_faces_by_face_walk(
            edges,
            points,
            min_face_nodes=min_face_nodes,
            max_face_nodes=max_face_nodes,
        )
        used_method = "face"
    elif method == "cycles":
        cells = reconstruct_triangles_by_cycles(edges, points)
        used_method = "cycles"
    elif method == "auto":
        cells = reconstruct_faces_by_face_walk(
            edges,
            points,
            min_face_nodes=min_face_nodes,
            max_face_nodes=max_face_nodes,
        )
        used_method = "face"
        if len(cells) == 0:
            cells = reconstruct_triangles_by_cycles(edges, points)
            used_method = "cycles"
    else:
        raise ValueError(f"Unknown method: {method}")

    if len(cells) == 0:
        raise ValueError(
            "No cells were reconstructed. The edge graph may not contain closed "
            "faces, or the points may not be a 2D embedding."
        )

    return cells, used_method


def reconstruct_triangles_from_edges(
    edges: np.ndarray,
    points: np.ndarray,
    *,
    method: str = "auto",
) -> tuple[np.ndarray, str]:
    cells, used_method = reconstruct_cells_from_edges(
        edges,
        points,
        method=method,
        min_face_nodes=3,
        max_face_nodes=3,
    )
    return np.asarray(cells, dtype=np.int32), used_method


def face_summary(cells: Sequence[Sequence[int]]) -> str:
    counts = {}
    for cell in cells:
        counts[len(cell)] = counts.get(len(cell), 0) + 1
    return ", ".join(f"{node_count}-node={count}" for node_count, count in sorted(counts.items()))


def cells_to_elems(
    cells: Sequence[Sequence[int]],
    *,
    max_nodes_per_elem: int,
    padding_value: int,
) -> np.ndarray:
    elems = np.full((len(cells), max_nodes_per_elem + 1), padding_value, dtype=np.int32)
    elems[:, 0] = 2
    for row_index, cell in enumerate(cells):
        if len(cell) > max_nodes_per_elem:
            raise ValueError(
                f"Cell has {len(cell)} nodes, but max_nodes_per_elem={max_nodes_per_elem}."
            )
        elems[row_index, 1 : 1 + len(cell)] = np.asarray(cell, dtype=np.int32)
    return elems


def process_one_folder(
    folder: Path,
    *,
    points_name: Optional[str],
    elements_name: str,
    edges_name: str,
    elems_name: str,
    padding_value: int,
    method: str,
    min_face_nodes: int,
    max_face_nodes: int,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int, str]:
    point_path = find_point_file(folder, points_name)
    elements_path = folder / elements_name
    edges_path = folder / edges_name
    elems_path = folder / elems_name

    if elements_path.exists():
        edge_source_path = elements_path
    elif edges_path.exists():
        edge_source_path = edges_path
    else:
        raise FileNotFoundError(f"Missing {elements_path} or existing {edges_path}.")

    if elems_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {elems_path}")
    if elements_path.exists() and edges_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {edges_path}")

    points = np.load(point_path)
    edge_like = np.load(edge_source_path)
    raw_edges = edge_endpoints(edge_like, padding_value=padding_value)
    edges, edge_stats = undirected_unique_edges(raw_edges, points.shape[0])
    try:
        cells, used_method = reconstruct_cells_from_edges(
            edges,
            points,
            method=method,
            min_face_nodes=min_face_nodes,
            max_face_nodes=max_face_nodes,
        )
    except ValueError as exc:
        raise ValueError(
            f"{folder}: {exc} "
            f"raw_edges={edge_stats['raw_edges']}, "
            f"undirected_edges={edge_stats['undirected_edges']}, "
            f"duplicate_or_reverse={edge_stats['duplicate_or_reverse']}, "
            f"self_loops={edge_stats['self_loops']}."
        ) from exc
    elems = cells_to_elems(
        cells,
        max_nodes_per_elem=max_face_nodes,
        padding_value=padding_value,
    )
    cells_summary = face_summary(cells)

    if dry_run:
        action = "rename" if elements_path.exists() else "use existing"
        print(
            f"[dry-run] {folder}: {action} {edge_source_path.name} -> {edges_path.name}; "
            f"write {elems_path.name}; method={used_method}, "
            f"raw_edges={edge_stats['raw_edges']}, undirected_edges={edge_stats['undirected_edges']}, "
            f"duplicate_or_reverse={edge_stats['duplicate_or_reverse']}, "
            f"self_loops={edge_stats['self_loops']}, cells={len(cells)} ({cells_summary})"
        )
        return edges.shape[0], len(cells), used_method

    if elements_path.exists():
        elements_path.replace(edges_path)
    np.save(elems_path, elems)

    print(
        f"{folder}: {edge_source_path.name} -> {edges_path.name}, "
        f"{elems_path.name} shape={elems.shape}; method={used_method}, "
        f"raw_edges={edge_stats['raw_edges']}, undirected_edges={edge_stats['undirected_edges']}, "
        f"duplicate_or_reverse={edge_stats['duplicate_or_reverse']}, "
        f"self_loops={edge_stats['self_loops']}, cells={len(cells)} ({cells_summary})"
    )
    return edges.shape[0], len(cells), used_method


def process_edge_folders(
    root_dir: Union[str, Path],
    *,
    start: int = 0,
    count: int = 300,
    points_name: Optional[str] = None,
    elements_name: str = "elements.npy",
    edges_name: str = "egdes.npy",
    elems_name: str = "elems.npy",
    padding_value: int = -1,
    method: str = "auto",
    min_face_nodes: int = 3,
    max_face_nodes: int = 4,
    skip_missing: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    root = Path(root_dir)
    total_edges = 0
    total_cells = 0
    processed = 0

    for folder_index in range(start, start + count):
        folder = root / str(folder_index)
        try:
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")
            edge_count, cell_count, _ = process_one_folder(
                folder,
                points_name=points_name,
                elements_name=elements_name,
                edges_name=edges_name,
                elems_name=elems_name,
                padding_value=padding_value,
                method=method,
                min_face_nodes=min_face_nodes,
                max_face_nodes=max_face_nodes,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        except FileNotFoundError as exc:
            if skip_missing:
                print(f"skip {folder}: {exc}")
                continue
            raise

        processed += 1
        total_edges += edge_count
        total_cells += cell_count

    print(
        f"Done. processed={processed}, total_edges={total_edges}, "
        f"total_cells={total_cells}"
    )


def main() -> None:
    args = parse_args()
    process_edge_folders(
        args.root_dir,
        start=args.start,
        count=args.count,
        points_name=args.points_name,
        elements_name=args.elements_name,
        edges_name=args.edges_name,
        elems_name=args.elems_name,
        padding_value=args.padding_value,
        method=args.method,
        min_face_nodes=args.min_face_nodes,
        max_face_nodes=args.max_face_nodes,
        skip_missing=args.skip_missing,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
