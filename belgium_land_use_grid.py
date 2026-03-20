#!/usr/bin/env python3
"""
Belgium Land Use Division — Grid-based Weighted Voronoi with Border Swapping.

Supports hierarchical categories: parent sectors contain sub-sectors on the map.
The algorithm runs in two levels — first allocating parent sectors across the
country, then sub-allocating children within each parent's region.
"""

import csv
import heapq
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt, label as ndlabel
import time
import warnings

warnings.filterwarnings("ignore")

RESOLUTION = 500  # meters per pixel
TARGET_CRS = "EPSG:3035"
AREA_TOLERANCE = 0.03


# ---------------------------------------------------------------------------
# Data loading and hierarchy
# ---------------------------------------------------------------------------

def load_categories(csv_path):
    """Load land-use categories from CSV (columns: name, area_km2, color, [parent])."""
    cats = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        has_parent = "parent" in (reader.fieldnames or [])
        for row in reader:
            parent_val = ""
            if has_parent:
                parent_val = (row.get("parent") or "").strip()
            cats.append({
                "name": row["name"].strip(),
                "area_km2": float(row["area_km2"]),
                "color": row["color"].strip(),
                "parent": parent_val or None,
            })
    return cats


def build_hierarchy(cats):
    """Build parent-child hierarchy from a flat category list.

    Returns (parent_groups, leaf_cats) where:
      parent_groups: list sorted by total_area desc; each dict has
          name, color, own_area, total_area, children, leaf_indices
      leaf_cats: flat list of all leaf categories for the final grid
    """
    parent_order = []
    children_map = {}
    parent_data = {}

    for cat in cats:
        if cat["parent"] is None:
            parent_order.append(cat["name"])
            parent_data[cat["name"]] = cat
            children_map.setdefault(cat["name"], [])
        else:
            children_map.setdefault(cat["parent"], []).append(cat)

    for name in parent_order:
        p = parent_data[name]
        children = children_map.get(name, [])
        p["_total_area"] = p["area_km2"]

    parent_order.sort(key=lambda n: parent_data[n]["_total_area"], reverse=True)

    parent_groups = []
    leaf_cats = []

    for name in parent_order:
        p = parent_data[name]
        children = children_map.get(name, [])
        children.sort(key=lambda c: c["area_km2"], reverse=True)

        children_total = sum(c["area_km2"] for c in children)
        own_area = p["area_km2"] - children_total

        group = {
            "name": name,
            "color": p["color"],
            "own_area": own_area,
            "total_area": p["area_km2"],
            "children": children,
            "leaf_indices": [],
        }

        # Children first — they receive interior seeds during sub-allocation
        for c in children:
            group["leaf_indices"].append(len(leaf_cats))
            leaf_cats.append({
                "name": c["name"],
                "area_km2": c["area_km2"],
                "color": c["color"],
                "parent_name": name,
                "is_parent_remaining": False,
            })

        # Parent remaining last — gets border seed, wraps around children
        if own_area > 0:
            group["leaf_indices"].append(len(leaf_cats))
            leaf_cats.append({
                "name": name,
                "area_km2": own_area,
                "color": p["color"],
                "parent_name": name,
                "is_parent_remaining": True,
            })

        parent_groups.append(group)

    return parent_groups, leaf_cats


# ---------------------------------------------------------------------------
# Phase 1: Rasterize Belgium
# ---------------------------------------------------------------------------

def load_and_rasterize(boundary_file="belgium.geojson"):
    """Load country boundary and rasterize to a boolean grid."""
    print(f"Phase 1: Rasterizing boundary from {boundary_file}...")
    t0 = time.time()

    gdf = gpd.read_file(boundary_file).to_crs(TARGET_CRS)
    belgium = gdf.geometry.unary_union

    minx, miny, maxx, maxy = belgium.bounds
    minx -= RESOLUTION
    miny -= RESOLUTION
    maxx += RESOLUTION
    maxy += RESOLUTION

    width = int((maxx - minx) / RESOLUTION) + 1
    height = int((maxy - miny) / RESOLUTION) + 1

    x_coords = minx + (np.arange(width) + 0.5) * RESOLUTION
    y_coords = maxy - (np.arange(height) + 0.5) * RESOLUTION

    polys = list(belgium.geoms) if isinstance(belgium, MultiPolygon) else [belgium]

    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    inside = np.zeros(len(points), dtype=bool)
    for poly in polys:
        path = MplPath(np.array(poly.exterior.coords))
        inside |= path.contains_points(points)
        for hole in poly.interiors:
            inside &= ~MplPath(np.array(hole.coords)).contains_points(points)

    mask = inside.reshape((height, width))

    n_pixels = int(mask.sum())
    pix_km2 = (RESOLUTION / 1000) ** 2
    print(f"  Grid {width}x{height}, {n_pixels} cells inside Belgium "
          f"({n_pixels * pix_km2:.0f} km², pixel={pix_km2} km²)")
    print(f"  Time: {time.time() - t0:.1f}s")

    tf = {"minx": minx, "maxy": maxy, "res": RESOLUTION}
    return mask, belgium, tf


# ---------------------------------------------------------------------------
# Phase 2: Seed placement (farthest-point sampling)
# ---------------------------------------------------------------------------

def place_seeds(mask, categories):
    """Place seeds using farthest-point sampling; most interior first."""
    print("Phase 2: Placing seeds (farthest-point sampling)...")
    t0 = time.time()

    n_cats = len(categories)
    h, w = mask.shape
    dist_border = distance_transform_edt(mask).astype(np.float32)

    rows, cols = np.mgrid[0:h, 0:w]
    seeds = []
    min_dist_seeds = np.full((h, w), np.inf, dtype=np.float32)

    for i in range(n_cats):
        if i == 0:
            score = dist_border.copy()
        else:
            score = min_dist_seeds.copy()
        score[~mask] = -np.inf
        idx = int(np.argmax(score.ravel()))
        r, c = np.unravel_index(idx, (h, w))
        seeds.append((int(r), int(c)))

        d = np.sqrt((rows - r) ** 2 + (cols - c) ** 2).astype(np.float32)
        min_dist_seeds = np.minimum(min_dist_seeds, d)

    seeds.sort(key=lambda s: dist_border[s[0], s[1]], reverse=True)

    for i, (r, c) in enumerate(seeds):
        print(f"  Seed {i:2d} ({categories[i]['name']:<22s}): "
              f"pixel ({r},{c}), border_dist={dist_border[r, c]:.0f}")
    print(f"  Time: {time.time() - t0:.1f}s")
    return seeds


# ---------------------------------------------------------------------------
# Phase 3: Weighted Voronoi (iterative weight adjustment)
# ---------------------------------------------------------------------------

def weighted_voronoi(mask, seeds, target_counts, max_iter=400):
    """Assign pixels via additively weighted Voronoi; iterate weights to match areas."""
    print("Phase 3: Weighted Voronoi iteration...")
    t0 = time.time()

    h, w = mask.shape
    n = len(seeds)

    rows, cols = np.mgrid[0:h, 0:w]
    dists = np.empty((n, h, w), dtype=np.float32)
    for i, (r, c) in enumerate(seeds):
        dists[i] = np.sqrt((rows - r) ** 2 + (cols - c) ** 2)

    weights = np.zeros(n, dtype=np.float64)
    total_inside = int(mask.sum())
    best_max_err = np.inf
    best_grid = None

    for it in range(max_iter):
        weighted = dists - weights[:, None, None].astype(np.float32)
        grid = weighted.argmin(axis=0).astype(np.int16)
        grid[~mask] = -1

        counts = np.array([(grid == i).sum() for i in range(n)])
        errors = np.where(target_counts > 0,
                          (counts - target_counts) / target_counts, 0.0)
        max_err = float(np.max(np.abs(errors)))

        if max_err < best_max_err:
            best_max_err = max_err
            best_grid = grid.copy()

        if it % 50 == 0:
            print(f"  Iter {it:3d}: max_error={max_err:.4f}")
        if max_err < 0.005:
            print(f"  Converged at iter {it} (max_error={max_err:.4f})")
            break

        lr = np.sqrt(target_counts.astype(np.float64) / np.pi) * 0.6
        for i in range(n):
            if target_counts[i] > 0:
                err_ratio = (target_counts[i] - counts[i]) / target_counts[i]
                weights[i] += lr[i] * err_ratio

    print(f"  Best max_error achieved: {best_max_err:.4f}")
    print(f"  Time: {time.time() - t0:.1f}s")
    return best_grid


# ---------------------------------------------------------------------------
# Phase 4: Connectivity fix
# ---------------------------------------------------------------------------

def fix_connectivity(grid, mask, seeds, n_cats):
    """Keep only the connected component containing each seed; reassign the rest."""
    print("Phase 4: Fixing connectivity...")
    t0 = time.time()

    h, w = grid.shape
    fixed = grid.copy()
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    total_disc = 0

    for ci in range(n_cats):
        sr, sc = seeds[ci]
        if fixed[sr, sc] != ci:
            ys, xs = np.where(fixed == ci)
            if len(ys) == 0:
                continue
            d2 = (ys - sr) ** 2 + (xs - sc) ** 2
            nearest = int(np.argmin(d2))
            sr, sc = int(ys[nearest]), int(xs[nearest])

        visited = np.zeros((h, w), dtype=bool)
        visited[sr, sc] = True
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and fixed[nr, nc] == ci:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        disc = (fixed == ci) & ~visited
        nd = int(disc.sum())
        if nd > 0:
            fixed[disc] = -1
            total_disc += nd

    unassigned = (fixed == -1) & mask
    nu = int(unassigned.sum())
    if nu > 0:
        q = deque()
        for r in range(h):
            for c in range(w):
                if fixed[r, c] >= 0:
                    for dr, dc in dirs4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and fixed[nr, nc] == -1 and mask[nr, nc]:
                            q.append((r, c))
                            break
        while q:
            r, c = q.popleft()
            cat = fixed[r, c]
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and fixed[nr, nc] == -1 and mask[nr, nc]:
                    fixed[nr, nc] = cat
                    q.append((nr, nc))

    remaining = int(((fixed == -1) & mask).sum())
    print(f"  Disconnected pixels: {total_disc}, reassigned, remaining unassigned: {remaining}")
    print(f"  Time: {time.time() - t0:.1f}s")
    return fixed


# ---------------------------------------------------------------------------
# Phase 5: Border pixel swapping
# ---------------------------------------------------------------------------

def is_removable(grid, r, c, h, w):
    """O(1) check: does removing (r,c) keep its region 4-connected?

    Uses the ring-arc test on the 8-neighborhood cycle.
    """
    cat = grid[r, c]
    cycle = [(-1, 0), (-1, 1), (0, 1), (1, 1),
             (1, 0), (1, -1), (0, -1), (-1, -1)]

    is_same = []
    for dr, dc in cycle:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            is_same.append(grid[nr, nc] == cat)
        else:
            is_same.append(False)

    n_same = sum(is_same)
    if n_same <= 1:
        return True

    transitions = 0
    for i in range(8):
        if is_same[i] != is_same[(i + 1) % 8]:
            transitions += 1
    return transitions <= 2


def find_border_coords(grid, mask):
    """Return (N,2) array of border pixel coordinates (fast numpy)."""
    h, w = grid.shape
    border = np.zeros((h, w), dtype=bool)
    g = grid.astype(np.int32)
    border[1:, :] |= (g[1:, :] != g[:-1, :]) & mask[1:, :] & (g[1:, :] >= 0) & (g[:-1, :] >= 0)
    border[:-1, :] |= (g[:-1, :] != g[1:, :]) & mask[:-1, :] & (g[:-1, :] >= 0) & (g[1:, :] >= 0)
    border[:, 1:] |= (g[:, 1:] != g[:, :-1]) & mask[:, 1:] & (g[:, 1:] >= 0) & (g[:, :-1] >= 0)
    border[:, :-1] |= (g[:, :-1] != g[:, 1:]) & mask[:, :-1] & (g[:, :-1] >= 0) & (g[:, 1:] >= 0)
    return np.argwhere(border)


def swap_borders(grid, mask, target_counts, categories, max_passes=500):
    """Iteratively swap border pixels to improve area accuracy."""
    print("Phase 5: Border pixel swapping...")
    t0 = time.time()

    h, w = grid.shape
    n = len(target_counts)
    counts = np.array([(grid == i).sum() for i in range(n)])
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for pn in range(max_passes):
        errors = np.where(target_counts > 0,
                          (counts - target_counts) / target_counts, 0.0)
        max_err = float(np.max(np.abs(errors)))

        if pn % 25 == 0:
            print(f"  Pass {pn:3d}: max_error={max_err:.4f}, "
                  f"worst={categories[int(np.argmax(np.abs(errors)))]['name']}")
        if max_err <= AREA_TOLERANCE:
            print(f"  Converged at pass {pn} (max_error={max_err:.4f})")
            break

        bc = find_border_coords(grid, mask)
        np.random.shuffle(bc)

        n_swapped = 0
        for idx in range(len(bc)):
            r, c = int(bc[idx, 0]), int(bc[idx, 1])
            src = int(grid[r, c])
            if src < 0:
                continue

            best_dst = -1
            best_improvement = 0.0
            tc = target_counts
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    dst = int(grid[nr, nc])
                    if dst >= 0 and dst != src:
                        eb = (abs(counts[src] - tc[src]) / max(tc[src], 1) +
                              abs(counts[dst] - tc[dst]) / max(tc[dst], 1))
                        ea = (abs(counts[src] - 1 - tc[src]) / max(tc[src], 1) +
                              abs(counts[dst] + 1 - tc[dst]) / max(tc[dst], 1))
                        improvement = eb - ea
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_dst = dst

            if best_dst < 0:
                continue

            if not is_removable(grid, r, c, h, w):
                continue

            grid[r, c] = best_dst
            counts[src] -= 1
            counts[best_dst] += 1
            n_swapped += 1

        if n_swapped == 0:
            print(f"  No swaps at pass {pn}, stopping.")
            break

    errors = np.where(target_counts > 0,
                      (counts - target_counts) / target_counts, 0.0)
    print(f"  Final max_error: {np.max(np.abs(errors)):.4f}")
    print(f"  Time: {time.time() - t0:.1f}s")
    return grid


# ---------------------------------------------------------------------------
# Allocate region (wraps phases 2–5 for any mask and category set)
# ---------------------------------------------------------------------------

def allocate_region(mask, categories, label=""):
    """Run the full allocation pipeline (phases 2–5) for a given mask."""
    n_cats = len(categories)
    if n_cats == 0:
        return np.full(mask.shape, -1, dtype=np.int16)
    if n_cats == 1:
        grid = np.full(mask.shape, -1, dtype=np.int16)
        grid[mask] = 0
        return grid

    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Allocating {n_cats} categories in {int(mask.sum())} pixels...")

    seeds = place_seeds(mask, categories)

    n_pixels = int(mask.sum())
    pix_km2 = (RESOLUTION / 1000) ** 2
    target_counts = np.array([c["area_km2"] / pix_km2 for c in categories])
    target_counts = target_counts * (n_pixels / target_counts.sum())
    target_counts = np.round(target_counts).astype(np.int64)
    diff = n_pixels - target_counts.sum()
    target_counts[0] += diff

    grid = weighted_voronoi(mask, seeds, target_counts)
    grid = fix_connectivity(grid, mask, seeds, n_cats)
    grid = swap_borders(grid, mask, target_counts, categories)

    return grid


def allocate_interior_islands(parent_mask, sub_cats, label=""):
    """Place children as compact circular islands in the parent interior.

    Each child is grown outward from the most interior available point via
    BFS sorted by Euclidean distance, producing nearly circular contiguous
    regions.  The parent-remaining (if any) fills whatever is left.
    """
    h, w = parent_mask.shape
    grid = np.full((h, w), -1, dtype=np.int16)
    prefix = f"[{label}] " if label else ""

    parent_rem_idx = None
    children = []
    for i, cat in enumerate(sub_cats):
        if cat.get("is_parent_remaining"):
            parent_rem_idx = i
        else:
            children.append((i, cat))

    if not children:
        if parent_rem_idx is not None:
            grid[parent_mask] = parent_rem_idx
        return grid

    n_total = int(parent_mask.sum())
    pix_km2 = (RESOLUTION / 1000) ** 2
    target_arr = np.array([c["area_km2"] / pix_km2 for c in sub_cats])
    target_arr = target_arr * (n_total / target_arr.sum())
    target_arr = np.round(target_arr).astype(np.int64)
    diff = n_total - target_arr.sum()
    if parent_rem_idx is not None:
        target_arr[parent_rem_idx] += diff
    else:
        target_arr[-1] += diff

    interiority = distance_transform_edt(parent_mask).astype(np.float32)
    available = parent_mask.copy()
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    children.sort(key=lambda x: x[1]["area_km2"], reverse=True)

    for _, (i, cat) in enumerate(children):
        n_pixels = int(target_arr[i])
        if n_pixels <= 0:
            continue

        score = interiority.copy()
        score[~available] = -np.inf
        if score.max() <= 0:
            break
        sr, sc = np.unravel_index(int(np.argmax(score.ravel())), (h, w))

        heap = [(0.0, sr, sc)]
        in_heap = np.zeros((h, w), dtype=bool)
        in_heap[sr, sc] = True
        count = 0

        while heap and count < n_pixels:
            d, r, c = heapq.heappop(heap)
            if not available[r, c]:
                continue
            grid[r, c] = i
            available[r, c] = False
            count += 1

            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w
                        and available[nr, nc] and not in_heap[nr, nc]):
                    in_heap[nr, nc] = True
                    nd = np.sqrt((nr - sr) ** 2 + (nc - sc) ** 2)
                    heapq.heappush(heap, (nd, nr, nc))

        child_dist = distance_transform_edt(~(grid == i)).astype(np.float32)
        interiority = np.minimum(interiority, child_dist)

        print(f"{prefix}{cat['name']}: {count} pixels "
              f"(target {n_pixels}, seed ({sr},{sc}))")

    remaining = available & parent_mask
    if remaining.any():
        if parent_rem_idx is not None:
            grid[remaining] = parent_rem_idx
            print(f"{prefix}{sub_cats[parent_rem_idx]['name']}: "
                  f"{int(remaining.sum())} pixels (remaining)")
        elif children:
            last_idx = children[-1][0]
            grid[remaining] = last_idx
            print(f"{prefix}{sub_cats[last_idx]['name']}: "
                  f"+{int(remaining.sum())} leftover pixels")

    return grid


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(grid, mask, categories):
    """Print contiguity and area accuracy for every category."""
    n_cats = len(categories)
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    pix_km2 = (RESOLUTION / 1000) ** 2
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    all_pass = True
    contiguous_count = 0
    area_count = 0

    print(f"{'Category':<25s} {'Area km²':>10s} {'Target':>10s} "
          f"{'Error%':>8s} {'Comp':>5s} {'Status':>8s}")
    print("-" * 75)

    for i, cat in enumerate(categories):
        cat_mask = (grid == i)
        actual = float(cat_mask.sum()) * pix_km2
        target = cat["area_km2"]
        err = (actual - target) / target * 100 if target > 0 else 0.0
        _, nc = ndlabel(cat_mask, structure=struct4)

        ok_area = abs(err) <= AREA_TOLERANCE * 100
        is_remaining = cat.get("is_parent_remaining", False)
        ok_cont = nc == 1 or (is_remaining and nc > 0)
        status = "PASS" if (ok_area and ok_cont) else "FAIL"
        if ok_cont:
            contiguous_count += 1
        if ok_area:
            area_count += 1
        if not (ok_area and ok_cont):
            all_pass = False

        print(f"{cat['name']:<25s} {actual:>10.1f} {target:>10.1f} "
              f"{err:>+7.2f}% {nc:>5d} {status:>8s}")

    print("-" * 75)
    print(f"Contiguous: {contiguous_count}/{n_cats}  |  "
          f"Area ±{AREA_TOLERANCE*100:.0f}%: {area_count}/{n_cats}  |  "
          f"Both: {'ALL PASS' if all_pass else 'FAIL'}")
    print("=" * 80)
    return all_pass


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(h):
    return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]


def _text_color(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 140 else "black"


def _find_label_position(cr, cc, mask, country_cr, country_cc, offset_px=30):
    """Walk outward from (cr, cc) until exiting *mask*, then add *offset_px*."""
    h, w = mask.shape
    dr = cr - country_cr
    dc = cc - country_cc
    length = max(np.sqrt(dr ** 2 + dc ** 2), 1.0)
    dr /= length
    dc /= length

    exit_step = 1
    for step in range(1, max(h, w)):
        nr = int(round(cr + dr * step))
        nc = int(round(cc + dc * step))
        if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
            exit_step = step
            break

    return cr + dr * (exit_step + offset_px), cc + dc * (exit_step + offset_px)


def _resolve_collisions(positions, min_gap, iterations=60, aspect=1.0):
    """Push (x, y) positions apart so no two label bounding boxes overlap.

    aspect: width/height ratio of label boxes.  When > 1 the horizontal gap
    is scaled up accordingly (AABB collision).  aspect=1.0 falls back to
    the original circular distance check for backward compatibility.
    """
    pos = [list(p) for p in positions]
    for _ in range(iterations):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]

                if aspect > 1.01:
                    gap_x = min_gap * aspect
                    overlap_x = gap_x - abs(dx)
                    overlap_y = min_gap - abs(dy)
                    if overlap_x > 0 and overlap_y > 0:
                        if overlap_x < overlap_y:
                            sign = 1.0 if dx >= 0 else -1.0
                            push = overlap_x / 2 * 1.1
                            pos[i][0] += push * sign
                            pos[j][0] -= push * sign
                        else:
                            sign = 1.0 if dy >= 0 else -1.0
                            push = overlap_y / 2 * 1.1
                            pos[i][1] += push * sign
                            pos[j][1] -= push * sign
                        moved = True
                else:
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    if 0 < dist < min_gap:
                        push = (min_gap - dist) / 2 * 1.1
                        nx, ny = dx / dist, dy / dist
                        pos[i][0] += push * nx
                        pos[i][1] += push * ny
                        pos[j][0] -= push * nx
                        pos[j][1] -= push * ny
                        moved = True
        if not moved:
            break
    return [(p[0], p[1]) for p in pos]


def _add_cc_by(fig, author=""):
    """Add a CC-BY 4.0 licence badge and author credit at the bottom of *fig*."""
    import os
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cc-by.png")
    lines = ["CC BY 4.0"]
    if author:
        parts = [s.strip() for s in author.split(",")]
        lines.extend(parts)
    label = "\n".join(lines)

    if os.path.isfile(icon_path):
        from matplotlib.offsetbox import OffsetImage, VPacker, HPacker, \
            TextArea, AnchoredOffsetbox
        icon = plt.imread(icon_path)
        icon_height_in = 0.35
        dpi = fig.dpi
        zoom = (icon_height_in * dpi) / icon.shape[0]
        im = OffsetImage(icon, zoom=zoom)
        txt = TextArea(label, textprops=dict(fontsize=11, color="#333333",
                                             ha="center", linespacing=1.4))
        pack = VPacker(children=[im, txt], align="center", pad=0, sep=3)
        ab = AnchoredOffsetbox(loc="lower right", child=pack,
                               bbox_to_anchor=(1.0, 0.0),
                               bbox_transform=fig.transFigure,
                               frameon=False, pad=0.3)
        fig.add_artist(ab)
    else:
        fig.text(0.99, 0.01, label,
                 fontsize=11, color="#555555", ha="right", va="bottom",
                 transform=fig.transFigure)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(grid, mask, country_geom, tf, leaf_cats, parent_groups,
              total_area, country_name="Belgium", small_threshold=0.02,
              filename="belgium_land_use_grid"):
    """Create a labelled land-use map with hierarchical parent/sub-sector display."""
    print("Visualization...")
    t0 = time.time()

    h, w = grid.shape
    res = tf["res"]

    # ---- parent index per pixel (for boundary drawing) ------------------
    parent_of = np.full((h, w), -1, dtype=np.int32)
    for pi, pg in enumerate(parent_groups):
        for li in pg["leaf_indices"]:
            parent_of[grid == li] = pi

    # ---- raster image ---------------------------------------------------
    img = np.ones((h, w, 3), dtype=np.float32)
    for i, cat in enumerate(leaf_cats):
        rgb = [int(cat["color"][1:3], 16) / 255,
               int(cat["color"][3:5], 16) / 255,
               int(cat["color"][5:7], 16) / 255]
        img[grid == i] = rgb
    img[~mask] = [1, 1, 1]

    fig, ax = plt.subplots(figsize=(16, 12))
    extent = [tf["minx"], tf["minx"] + w * res,
              tf["maxy"] - h * res, tf["maxy"]]
    ax.imshow(img, extent=extent, origin="upper", interpolation="nearest")

    gpd.GeoDataFrame(geometry=[country_geom], crs=TARGET_CRS) \
       .boundary.plot(ax=ax, color="black", linewidth=2)

    # ---- draw clean vector borders between parent sectors ---------------
    has_hierarchy = any(len(pg["children"]) > 0 for pg in parent_groups)
    if has_hierarchy:
        parent_float = parent_of.astype(np.float64)
        parent_float[~mask] = np.nan
        xs = np.linspace(extent[0], extent[1], w)
        ys = np.linspace(extent[3], extent[2], h)
        levels = np.arange(0.5, len(parent_groups))
        ax.contour(xs, ys, parent_float, levels=levels,
                   colors='#333333', linewidths=1.2)

    # ---- country centroid (grid coords) ---------------------------------
    ys_all, xs_all = np.where(mask)
    country_cr = float(ys_all.mean())
    country_cc = float(xs_all.mean())

    # ---- classify labels as large (inline) or small (external) ----------
    large_labels = []
    small_labels = []

    for i, cat in enumerate(leaf_cats):
        cat_mask = (grid == i)
        if not cat_mask.any():
            continue
        ys, xs = np.where(cat_mask)
        cr, cc = float(ys.mean()), float(xs.mean())
        cx = tf["minx"] + (cc + 0.5) * res
        cy = tf["maxy"] - (cr + 0.5) * res
        area_frac = cat["area_km2"] / total_area

        if area_frac >= small_threshold:
            large_labels.append(dict(cat=cat, cx=cx, cy=cy, frac=area_frac))
        else:
            lr, lc = _find_label_position(
                cr, cc, mask, country_cr, country_cc, offset_px=40)
            lx = tf["minx"] + (lc + 0.5) * res
            ly = tf["maxy"] - (lr + 0.5) * res
            small_labels.append(dict(cat=cat, cx=cx, cy=cy,
                                     lx=lx, ly=ly, frac=area_frac))

    # ---- draw large labels (inside the regions) -------------------------
    for lb in large_labels:
        fs = max(10, min(17, 10 + lb["frac"] * 35))
        ax.text(lb["cx"], lb["cy"], lb["cat"]["name"],
                ha="center", va="center", fontsize=fs, fontweight="bold",
                color=_text_color(lb["cat"]["color"]))

    # ---- resolve collisions for small external labels -------------------
    if small_labels:
        raw_pos = [(s["lx"], s["ly"]) for s in small_labels]
        span = max(extent[1] - extent[0], extent[3] - extent[2])
        gap = span * 0.055
        resolved = _resolve_collisions(raw_pos, min_gap=gap, iterations=200,
                                       aspect=2.5)

        y_max_label = extent[3] + span * 0.02
        resolved = [(lx, min(ly, y_max_label)) for lx, ly in resolved]
        resolved = _resolve_collisions(resolved, min_gap=gap, iterations=200,
                                       aspect=2.5)

        mask_dist = distance_transform_edt(~mask).astype(np.float32)
        min_margin_px = 20
        for k in range(len(resolved)):
            lx, ly = resolved[k]
            lc_g = (lx - extent[0]) / res
            lr_g = (extent[3] - ly) / res
            lr_i, lc_i = int(round(lr_g)), int(round(lc_g))
            if (0 <= lr_i < h and 0 <= lc_i < w
                    and mask_dist[lr_i, lc_i] < min_margin_px):
                push = min_margin_px - mask_dist[lr_i, lc_i] + 5
                vec_r = lr_g - country_cr
                vec_c = lc_g - country_cc
                vlen = max(np.sqrt(vec_r ** 2 + vec_c ** 2), 1.0)
                resolved[k] = (lx + (vec_c / vlen) * push * res,
                                ly - (vec_r / vlen) * push * res)

        for k, sl in enumerate(small_labels):
            lx, ly = resolved[k]
            cat = sl["cat"]
            ax.annotate(
                cat["name"], xy=(sl["cx"], sl["cy"]), xytext=(lx, ly),
                fontsize=13, fontweight="bold",
                color=_text_color(cat["color"]),
                arrowprops=dict(arrowstyle="-", color="#444444", lw=1.0,
                                connectionstyle="arc3,rad=0.12"),
                bbox=dict(boxstyle="round,pad=0.3", fc=cat["color"],
                          alpha=0.92, ec="#444444", lw=0.6),
                ha="center", va="center", zorder=6,
            )
            ax.plot(sl["cx"], sl["cy"], "o", color=cat["color"],
                    markersize=6, markeredgecolor="#333", markeredgewidth=1,
                    zorder=7)

    # ---- hierarchical legend --------------------------------------------
    handles = []
    for pg in parent_groups:
        has_children = len(pg["children"]) > 0
        if not has_children:
            lc = leaf_cats[pg["leaf_indices"][0]]
            handles.append(mpatches.Patch(
                facecolor=lc["color"],
                label=f"{lc['name']} ({lc['area_km2']:,.0f} km²)"))
        else:
            alpha = 0.92 if pg["own_area"] > 0 else 0.35
            handles.append(mpatches.Patch(
                facecolor=pg["color"], alpha=alpha,
                edgecolor=pg["color"], linewidth=1.5,
                label=f"{pg['name']} ({pg['total_area']:,.0f} km²)"))
            for li in pg["leaf_indices"]:
                lc = leaf_cats[li]
                if lc.get("is_parent_remaining"):
                    continue
                handles.append(mpatches.Patch(
                    facecolor=lc["color"],
                    label=f"    {lc['name']} ({lc['area_km2']:,.0f} km²)"))

    leg = ax.legend(handles=handles, loc="upper left", fontsize=9,
                    title="Land Use", title_fontsize=12,
                    bbox_to_anchor=(1.01, 1), framealpha=0.95)
    for text in leg.get_texts():
        if not text.get_text().startswith("    "):
            text.set_fontweight("bold")

    ax.set_title(f"Land Use — {country_name}",
                 fontsize=24, fontweight="bold", pad=35)
    ax.set_axis_off()

    _add_cc_by(fig, author="Sylvain Quoilin, University of Liège")

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved {filename}.png")
    print(f"  Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Optional GeoJSON export
# ---------------------------------------------------------------------------

def export_geojson(grid, mask, tf, categories, filename="belgium_land_use_grid"):
    """Convert grid to polygon GeoJSON (one feature per category)."""
    print("Exporting GeoJSON...")
    t0 = time.time()
    res = tf["res"]
    h, w = grid.shape

    features = []
    for i, cat in enumerate(categories):
        cat_mask = (grid == i)
        if not cat_mask.any():
            continue

        ys, xs = np.where(cat_mask)
        boxes = []
        for y, x in zip(ys, xs):
            x0 = tf["minx"] + x * res
            y0 = tf["maxy"] - (y + 1) * res
            boxes.append(box(x0, y0, x0 + res, y0 + res))

        merged = unary_union(boxes)
        features.append({
            "category": cat["name"],
            "area_km2": float(cat_mask.sum()) * (res / 1000) ** 2,
            "target_km2": cat["area_km2"],
            "color": cat["color"],
            "geometry": merged,
        })
        print(f"  {cat['name']}: vectorized ({len(boxes)} cells)")

    gdf = gpd.GeoDataFrame(features, crs=TARGET_CRS)
    gdf.to_file(f"{filename}.geojson", driver="GeoJSON")
    print(f"  Saved {filename}.geojson")
    print(f"  Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path="belgium_land_use.csv", boundary_file="belgium.geojson",
         country_name="Belgium", output_prefix="belgium_land_use_grid"):
    all_cats = load_categories(csv_path)
    parent_groups, leaf_cats = build_hierarchy(all_cats)
    n_parents = len(parent_groups)
    n_leaves = len(leaf_cats)
    total_area = sum(c["area_km2"] for c in leaf_cats)

    print("=" * 70)
    print(f"  {country_name.upper()} LAND USE — Hierarchical Grid Allocation")
    print(f"  {n_parents} parent sectors, {n_leaves} leaf categories")
    print("=" * 70)
    t_total = time.time()

    mask, country_geom, tf = load_and_rasterize(boundary_file)

    # --- Level 1: allocate parent sectors across the country ---
    parent_cats = [{"name": pg["name"], "area_km2": pg["total_area"],
                    "color": pg["color"]} for pg in parent_groups]

    print(f"\n{'=' * 60}")
    print("LEVEL 1: Parent sector allocation")
    print(f"{'=' * 60}")
    parent_grid = allocate_region(mask, parent_cats, label="L1")

    # --- Level 2: sub-allocate within each parent ---
    print(f"\n{'=' * 60}")
    print("LEVEL 2: Sub-sector allocation within parents")
    print(f"{'=' * 60}")

    final_grid = np.full_like(parent_grid, -1)

    for pi, pg in enumerate(parent_groups):
        parent_mask = (parent_grid == pi) & mask
        leaves = pg["leaf_indices"]

        if len(leaves) == 1:
            final_grid[parent_mask] = leaves[0]
            print(f"\n  {pg['name']}: standalone ({int(parent_mask.sum())} pixels)")
        else:
            sub_cats = [leaf_cats[li] for li in leaves]
            has_remaining = any(c.get("is_parent_remaining") for c in sub_cats)
            if has_remaining:
                sub_grid = allocate_interior_islands(
                    parent_mask, sub_cats, label=pg["name"])
            else:
                sub_grid = allocate_region(
                    parent_mask, sub_cats, label=pg["name"])
            for si, li in enumerate(leaves):
                final_grid[(sub_grid == si) & parent_mask] = li

    # --- Validate and visualize ---
    passed = validate(final_grid, mask, leaf_cats)

    visualize(final_grid, mask, country_geom, tf, leaf_cats, parent_groups,
              total_area, country_name=country_name, filename=output_prefix)

    total = time.time() - t_total
    print(f"\nTotal runtime: {total:.1f}s")
    if passed:
        print(f"SUCCESS: All {n_leaves} leaf categories pass both contiguity "
              "and area tolerance!")
    else:
        print("Some categories did not pass — see validation above.")


if __name__ == "__main__":
    import sys, os
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "belgium_land_use.csv"
    prefix = os.path.splitext(csv_file)[0]
    main(csv_path=csv_file, output_prefix=prefix)
