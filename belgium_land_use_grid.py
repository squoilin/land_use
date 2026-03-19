#!/usr/bin/env python3
"""
Belgium Land Use Division — Grid-based Weighted Voronoi with Border Swapping.

Instead of working with ~19,795 irregular statistical sectors, this approach
rasterizes Belgium onto a regular grid (~120K cells at 500m). Adjacency is
implicit, region growing is trivial BFS, and area fine-tuning is fast pixel
swapping with O(1) contiguity checks.
"""

import csv
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


def load_categories(csv_path):
    """Load land-use categories from a CSV file (columns: name, area_km2, color)."""
    cats = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cats.append({
                "name": row["name"].strip(),
                "area_km2": int(row["area_km2"]),
                "color": row["color"].strip(),
            })
    cats.sort(key=lambda c: c["area_km2"], reverse=True)
    return cats


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

            # Find best adjacent category to receive this pixel
            # Use relative (percentage) errors so small categories are prioritised
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
        err = (actual - target) / target * 100
        _, nc = ndlabel(cat_mask, structure=struct4)

        ok_area = abs(err) <= AREA_TOLERANCE * 100
        ok_cont = nc == 1
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
# Visualization
# ---------------------------------------------------------------------------

def _hex_to_rgb(h):
    return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]


def _text_color(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 140 else "black"


def _find_label_position(cr, cc, mask, country_cr, country_cc, offset_px=30):
    """Walk outward from (cr, cc) until exiting *mask*, then add *offset_px*.

    Generic: works for any country shape.  Returns (label_r, label_c) in grid
    coordinates.
    """
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


def _resolve_collisions(positions, min_gap, iterations=60):
    """Push (x, y) positions apart so no two are closer than *min_gap*."""
    pos = [list(p) for p in positions]
    for _ in range(iterations):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
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
    """Add a CC-BY 4.0 licence badge and author credit at the bottom of *fig*.

    Uses the official CC-BY icon from a local PNG if available, otherwise falls
    back to a text-only notice.
    """
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
        icon_height_in = 0.22
        dpi = fig.dpi
        zoom = (icon_height_in * dpi) / icon.shape[0]
        im = OffsetImage(icon, zoom=zoom)
        txt = TextArea(label, textprops=dict(fontsize=8, color="#333333",
                                             ha="center", linespacing=1.4))
        pack = VPacker(children=[im, txt], align="center", pad=0, sep=3)
        ab = AnchoredOffsetbox(loc="lower right", child=pack,
                               bbox_to_anchor=(1.0, 0.0),
                               bbox_transform=fig.transFigure,
                               frameon=False, pad=0.3)
        fig.add_artist(ab)
    else:
        fig.text(0.99, 0.01, label,
                 fontsize=8, color="#555555", ha="right", va="bottom",
                 transform=fig.transFigure)


def visualize(grid, mask, country_geom, tf, categories, total_area,
              country_name="Belgium", small_threshold=0.02,
              filename="belgium_land_use_grid"):
    """Create a labelled land-use map.  Generic for any country."""
    print("Phase 6: Visualization...")
    t0 = time.time()

    h, w = grid.shape
    res = tf["res"]

    # ---- raster image -------------------------------------------------
    img = np.ones((h, w, 3), dtype=np.float32)
    for i, cat in enumerate(categories):
        col = cat["color"]
        rgb = [int(col[1:3], 16) / 255, int(col[3:5], 16) / 255,
               int(col[5:7], 16) / 255]
        img[grid == i] = rgb
    img[~mask] = [1, 1, 1]

    fig, ax = plt.subplots(figsize=(16, 12))
    extent = [tf["minx"], tf["minx"] + w * res,
              tf["maxy"] - h * res, tf["maxy"]]
    ax.imshow(img, extent=extent, origin="upper", interpolation="nearest")

    gpd.GeoDataFrame(geometry=[country_geom], crs=TARGET_CRS) \
       .boundary.plot(ax=ax, color="black", linewidth=2)

    # ---- country centroid (grid coords) -------------------------------
    ys_all, xs_all = np.where(mask)
    country_cr = float(ys_all.mean())
    country_cc = float(xs_all.mean())

    # ---- classify labels as large (inline) or small (external) --------
    large_labels = []
    small_labels = []

    for i, cat in enumerate(categories):
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
                cr, cc, mask, country_cr, country_cc, offset_px=28)
            lx = tf["minx"] + (lc + 0.5) * res
            ly = tf["maxy"] - (lr + 0.5) * res
            small_labels.append(dict(cat=cat, cx=cx, cy=cy,
                                     lx=lx, ly=ly, frac=area_frac))

    # ---- draw large labels (inside the regions) -----------------------
    for lb in large_labels:
        fs = max(10, min(17, 10 + lb["frac"] * 35))
        ax.text(lb["cx"], lb["cy"], lb["cat"]["name"],
                ha="center", va="center", fontsize=fs, fontweight="bold",
                color=_text_color(lb["cat"]["color"]))

    # ---- resolve collisions for small external labels -----------------
    if small_labels:
        raw_pos = [(s["lx"], s["ly"]) for s in small_labels]
        span = max(extent[1] - extent[0], extent[3] - extent[2])
        resolved = _resolve_collisions(raw_pos, min_gap=span * 0.055)

        # Clamp labels so they stay below the title
        y_max_label = extent[3] + span * 0.02
        resolved = [(lx, min(ly, y_max_label)) for lx, ly in resolved]

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

    # ---- legend -------------------------------------------------------
    legend_patches = [
        mpatches.Patch(color=cat["color"],
                       label=f"{cat['name']} ({cat['area_km2']:,} km²)")
        for cat in categories
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=10,
              title="Land Use", title_fontsize=12,
              bbox_to_anchor=(1.01, 1), framealpha=0.95)

    ax.set_title(f"Land Use — {country_name}",
                 fontsize=24, fontweight="bold", pad=35)
    ax.set_axis_off()

    # ---- CC-BY licence + author ------------------------------------------
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
    print("Phase 7: Exporting GeoJSON...")
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
    categories = load_categories(csv_path)
    n_cats = len(categories)
    total_area = sum(c["area_km2"] for c in categories)

    print("=" * 70)
    print(f"  {country_name.upper()} LAND USE — Grid-based Weighted Voronoi")
    print("=" * 70)
    t_total = time.time()

    mask, country_geom, tf = load_and_rasterize(boundary_file)

    seeds = place_seeds(mask, categories)

    n_pixels = int(mask.sum())
    pix_km2 = (RESOLUTION / 1000) ** 2
    target_counts = np.array([c["area_km2"] / pix_km2 for c in categories])
    target_counts = target_counts * (n_pixels / target_counts.sum())
    target_counts = np.round(target_counts).astype(np.int64)
    diff = n_pixels - target_counts.sum()
    target_counts[0] += diff
    print(f"\nPixels inside Belgium: {n_pixels}")
    print(f"Target counts sum:     {int(target_counts.sum())}")

    grid = weighted_voronoi(mask, seeds, target_counts)

    grid = fix_connectivity(grid, mask, seeds, n_cats)

    grid = swap_borders(grid, mask, target_counts, categories)

    passed = validate(grid, mask, categories)

    visualize(grid, mask, country_geom, tf, categories, total_area,
              country_name=country_name, filename=output_prefix)

    total = time.time() - t_total
    print(f"\nTotal runtime: {total:.1f}s")
    if passed:
        print(f"SUCCESS: All {n_cats} categories pass both contiguity and "
              "area tolerance!")
    else:
        print("Some categories did not pass — see validation above.")


if __name__ == "__main__":
    main()
