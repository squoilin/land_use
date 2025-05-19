import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union, split
from scipy.spatial import Voronoi
from PIL import Image
import requests
import random
import os
import skimage.morphology
from skimage.draw import polygon as skpolygon
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from collections import deque

# Land use data (labels, values, colors)
land_use = [
    ("Agriculture", 44, "#e1cd87"),
    ("Forest", 22, "#7fc97f"),
    ("Urban", 13, "#f0027f"),
    ("Industry", 5, "#beaed4"),
    ("Water", 2, "#386cb0"),
    ("Wetlands", 2, "#66c2a5"),
    ("Other", 12, "#bdbdbd"),
]
labels, values, colors = zip(*land_use)
values = np.array(values)
areas = values / values.sum()
N = len(land_use)

BELGIUM_GEOJSON = "belgium.geojson"
COUNTRIES_GEOJSON = "countries.geojson"

# Download Belgium boundary if not present
def download_belgium():
    if not os.path.exists(BELGIUM_GEOJSON):
        if not os.path.exists(COUNTRIES_GEOJSON):
            url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
            r = requests.get(url)
            with open(COUNTRIES_GEOJSON, "wb") as f:
                f.write(r.content)
        gdf = gpd.read_file(COUNTRIES_GEOJSON)
        belgium = gdf[gdf['name'] == 'Belgium']
        belgium.to_file(BELGIUM_GEOJSON, driver="GeoJSON")

def generate_random_points_within(poly, num_points):
    minx, miny, maxx, maxy = poly.bounds
    points = []
    attempts = 0
    while len(points) < num_points and attempts < num_points * 100:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(random_point):
            points.append(random_point)
        attempts += 1
    return points

def voronoi_finite_polygons_2d(vor, radius=None):
    # Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.
    # Adapted from https://stackoverflow.com/a/20678647
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max()*2
    # Map ridge vertices to ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct each region
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue
            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)

def lloyd_relaxation(points, poly, iterations=10):
    for _ in range(iterations):
        vor = Voronoi(points)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        new_points = []
        for region in regions:
            polygon = Polygon(vertices[region])
            clipped = polygon.intersection(poly)
            if not clipped.is_empty and clipped.geom_type == 'Polygon':
                new_points.append(np.array(clipped.centroid))
            elif not clipped.is_empty and clipped.geom_type == 'MultiPolygon':
                # Use the largest polygon in the multipolygon
                largest = max(clipped.geoms, key=lambda g: g.area)
                new_points.append(np.array(largest.centroid))
            else:
                # If the cell is empty, keep the old point
                new_points.append(points[len(new_points)])
        points = np.array([[p[0], p[1]] for p in new_points])
    return points

def rasterize_polygon(poly, shape):
    # Rasterize the polygon to a mask
    minx, miny, maxx, maxy = poly.bounds
    x = np.linspace(minx, maxx, shape[1])
    y = np.linspace(miny, maxy, shape[0])
    xv, yv = np.meshgrid(x, y)
    coords = np.vstack((xv.flatten(), yv.flatten())).T
    mask = np.array([poly.contains(Point(pt)) for pt in coords]).reshape(shape)
    return mask, (minx, miny, maxx, maxy)

def pick_random_seeds(mask, n):
    ys, xs = np.where(mask)
    idx = np.random.choice(len(xs), n, replace=False)
    return list(zip(ys[idx], xs[idx]))

def grow_regions(mask, areas):
    n = len(areas)
    region_map = np.zeros_like(mask, dtype=int) - 1
    seeds = pick_random_seeds(mask, n)
    frontiers = [[seed] for seed in seeds]
    region_sizes = [1 for _ in range(n)]
    for i, (y, x) in enumerate(seeds):
        region_map[y, x] = i
    total_pixels = mask.sum()
    target_pixels = (areas * total_pixels).astype(int)
    # Ensure sum matches
    target_pixels[-1] = total_pixels - target_pixels[:-1].sum()
    # Grow regions
    while any(region_sizes[i] < target_pixels[i] for i in range(n)):
        for i in range(n):
            if region_sizes[i] >= target_pixels[i]:
                continue
            if not frontiers[i]:
                continue
            y, x = frontiers[i].pop(0)
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    if mask[ny, nx] and region_map[ny, nx] == -1:
                        region_map[ny, nx] = i
                        region_sizes[i] += 1
                        frontiers[i].append((ny, nx))
                        if region_sizes[i] >= target_pixels[i]:
                            break
    return region_map

def watershed_regions(mask, n):
    # Distance transform for mask
    distance = ndi.distance_transform_edt(mask)
    # Place N random seeds inside the mask
    ys, xs = np.where(mask)
    idx = np.random.choice(len(xs), n, replace=False)
    markers = np.zeros_like(mask, dtype=int)
    for i, (y, x) in enumerate(zip(ys[idx], xs[idx])):
        markers[y, x] = i + 1  # watershed expects markers > 0
    # Watershed segmentation
    labels = watershed(-distance, markers, mask=mask)
    return labels - 1  # make labels 0-based

# Recursive polygon splitting by area

def split_polygon_by_area(polygon, areas, axis=0):
    """
    Recursively split the polygon into len(areas) contiguous sub-polygons,
    each with area proportional to areas (must sum to 1).
    axis: 0 for vertical, 1 for horizontal split (alternates at each recursion).
    Returns: list of sub-polygons
    """
    if len(areas) == 1:
        return [polygon]
    # Find the split value (fraction of area)
    total_area = polygon.area
    area1 = areas[0] * total_area
    # Find the split line
    minx, miny, maxx, maxy = polygon.bounds
    # Binary search for the split location
    lo, hi = (minx, maxx) if axis == 0 else (miny, maxy)
    for _ in range(30):
        mid = (lo + hi) / 2
        if axis == 0:
            line = LineString([(mid, miny-1), (mid, maxy+1)])
        else:
            line = LineString([(minx-1, mid), (maxx+1, mid)])
        splitted = split(polygon, line)
        # Filter only valid polygons
        splitted_polys = [g for g in splitted.geoms if g.is_valid and g.area > 1e-8]
        if len(splitted_polys) < 2:
            # Try a different split
            if axis == 0:
                lo = mid
            else:
                hi = mid
            continue
        p1, p2 = splitted_polys[0], unary_union(splitted_polys[1:])
        if p1.area > area1:
            hi = mid
        else:
            lo = mid
        if abs(p1.area - area1) < 1e-4 * total_area:
            break
    # Assign the smaller area to the first region
    if p1.area > p2.area:
        p1, p2 = p2, p1
    # Recursively split the rest
    return [p1] + split_polygon_by_area(p2, areas[1:], 1-axis)

def efficient_region_grow(mask, areas, progress_step=0.05, max_iter=1_000_000):
    n = len(areas)
    region_map = np.full(mask.shape, -1, dtype=int)
    ys, xs = np.where(mask)
    total_pixels = len(xs)
    target_pixels = (areas * total_pixels).astype(int)
    target_pixels[-1] = total_pixels - target_pixels[:-1].sum()  # ensure sum matches
    # Sort regions by target area (descending)
    sort_idx = np.argsort(-target_pixels)
    target_pixels_sorted = target_pixels[sort_idx]
    n = len(target_pixels_sorted)
    idx = np.random.choice(total_pixels, n, replace=False)
    seeds = list(zip(ys[idx], xs[idx]))
    queues = [deque([seed]) for seed in seeds]
    region_sizes = [1 for _ in range(n)]
    for i, (y, x) in enumerate(seeds):
        region_map[y, x] = i
    assigned = n
    last_progress = 0
    print(f"Growing regions: 0% ({assigned}/{total_pixels})", flush=True)
    iter_count = 0
    while assigned < total_pixels and iter_count < max_iter:
        iter_count += 1
        assigned_this_iter = 0
        for i in range(n):  # grow in sorted order
            if region_sizes[i] >= target_pixels_sorted[i]:
                continue
            if not queues[i]:
                continue
            y, x = queues[i].popleft()
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    if mask[ny, nx] and region_map[ny, nx] == -1:
                        region_map[ny, nx] = i
                        region_sizes[i] += 1
                        queues[i].append((ny, nx))
                        assigned += 1
                        assigned_this_iter += 1
                        progress = assigned / total_pixels
                        if progress - last_progress >= progress_step:
                            print(f"Growing regions: {int(progress*100)}% ({assigned}/{total_pixels})", flush=True)
                            last_progress = progress
                        if region_sizes[i] >= target_pixels_sorted[i]:
                            break
        if iter_count % 1000 == 0:
            active_frontiers = sum(len(q) for q in queues)
            print(f"Iteration {iter_count}: assigned={assigned}/{total_pixels}, active frontiers={active_frontiers}", flush=True)
        if assigned_this_iter == 0:
            print(f"No progress in iteration {iter_count}. Breaking to avoid infinite loop.", flush=True)
            break
    print(f"Growing regions: 100% ({assigned}/{total_pixels})", flush=True)
    # Remap region indices to original order
    remap = np.argsort(sort_idx)
    region_map_remapped = np.full_like(region_map, -1)
    for i in range(n):
        region_map_remapped[region_map == i] = remap[i]
    return region_map_remapped

def find_boundary_pixels(region_map):
    # Returns a list of (y, x) for pixels on a region boundary
    boundary = np.zeros_like(region_map, dtype=bool)
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        shifted = np.roll(region_map, shift=(dy, dx), axis=(0, 1))
        boundary |= (region_map != shifted)
    boundary &= (region_map != -1)
    return np.argwhere(boundary)

def is_contiguous(region_map, region_label, y, x):
    # Check if removing (y, x) from region_label disconnects the region
    # Simple check: after removal, does at least one neighbor remain with the same label?
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y+dy, x+dx
        if 0 <= ny < region_map.shape[0] and 0 <= nx < region_map.shape[1]:
            if region_map[ny, nx] == region_label:
                return True
    return False

def pixel_exchange(region_map, target_pixels, max_iters=10):
    N = len(target_pixels)
    for it in range(max_iters):
        improved = False
        area = np.array([np.sum(region_map == i) for i in range(N)])
        area_error = area - target_pixels
        total_error = np.sum(np.abs(area_error))
        boundary_pixels = find_boundary_pixels(region_map)
        np.random.shuffle(boundary_pixels)  # randomize order for fairness
        for y, x in boundary_pixels:
            label = region_map[y, x]
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < region_map.shape[0] and 0 <= nx < region_map.shape[1]:
                    nlabel = region_map[ny, nx]
                    if nlabel == -1 or nlabel == label:
                        continue
                    # Would swapping (y,x) to nlabel reduce total error?
                    new_area = area.copy()
                    new_area[label] -= 1
                    new_area[nlabel] += 1
                    new_error = np.sum(np.abs(new_area - target_pixels))
                    if new_error < total_error and is_contiguous(region_map, label, y, x):
                        region_map[y, x] = nlabel
                        area = new_area
                        area_error = area - target_pixels
                        total_error = new_error
                        improved = True
                        break  # Only one swap per pixel per iteration
        print(f"Pixel exchange iter {it+1}: total area error {total_error}")
        if not improved:
            print("No further improvement in pixel exchange.")
            break
    return region_map

def main():
    download_belgium()
    belgium = gpd.read_file(BELGIUM_GEOJSON)
    poly = belgium.geometry.values[0]
    if isinstance(poly, MultiPolygon):
        poly = unary_union(poly)

    # Rasterize Belgium
    shape = (500, 500)  # Higher resolution for better quality
    mask, (minx, miny, maxx, maxy) = rasterize_polygon(poly, shape)

    # Normalize areas
    norm_areas = np.array(values) / np.sum(values)
    total_pixels = np.sum(mask)
    target_pixels = (norm_areas * total_pixels).astype(int)
    target_pixels[-1] = total_pixels - target_pixels[:-1].sum()

    # Try multiple attempts and keep the best
    best_error = float('inf')
    best_region_map = None
    best_actuals = None
    attempts = 10
    for attempt in range(attempts):
        print(f"\nAttempt {attempt+1}/{attempts}")
        region_map = efficient_region_grow(mask, norm_areas)
        # Post-process with pixel exchange
        region_map = pixel_exchange(region_map, target_pixels, max_iters=10)
        actuals = [np.sum(region_map == i) for i in range(N)]
        error = sum(abs(a - t) for a, t in zip(actuals, target_pixels))
        print(f"Total area error: {error}")
        if error < best_error:
            best_error = error
            best_region_map = region_map.copy()
            best_actuals = actuals.copy()
        if error == 0:
            break

    region_map = best_region_map
    print("\nBest result summary:")
    print(f"{'Label':<12} {'Target':>8} {'Actual':>8} {'Error %':>8}")
    all_present = True
    for i, (label, target, actual) in enumerate(zip(labels, target_pixels, best_actuals)):
        error = 100 * (actual - target) / target if target > 0 else 0
        print(f"{label:<12} {target:8d} {actual:8d} {error:8.2f}")
        if actual == 0:
            all_present = False
    if not all_present:
        print("WARNING: Some regions are missing from the output!", flush=True)
    else:
        print("All regions are present.", flush=True)

    # Create RGB image
    img = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255
    for i, color in enumerate(colors):
        rgb = tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        img[region_map == i] = rgb

    # Save as PNG
    pil_img = Image.fromarray(img)
    pil_img.save("belgium_land_use_map.png")
    print("Belgium land use map generated as belgium_land_use_map.png, size:", pil_img.size)

    # Overlay Belgium border for clarity
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img, extent=(minx, maxx, miny, maxy), origin='lower')
    gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="black", linewidth=2)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("belgium_land_use_map_with_border.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Belgium land use map with border saved as belgium_land_use_map_with_border.png")

if __name__ == "__main__":
    main() 