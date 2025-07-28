We want to build a python script that divides any country (in our case belgium) into regions whose area is pre-determined. 
- The script should be written in python, and tested using the existing belgium_land_use_env conda environment
- The algorithm will most likely be iterative, but it should be computationally efficient. Output should be provided in stdout to follow the progress.
- It is important to consider different solutions and available libraries before selecting the most promising one
- A check function should be executed after the algorithm to verify the all regions are present and that their areas are correct
- The input data for the areas is provided in the markdown table below. When there are subcategories, they should be represented and the general category should be renamed into "category_name (others)" to show that it does not include the subcategory


### ✅ Flattened Land Use Table (No Aggregation)

| Category                               | Area (km²) | % of Total |
| -------------------------------------- | ---------- | ---------- |
| Agricultural land (others)             | 12,803     | 41.7%      |
| Energy crops                           | 700        | 2.3%       |
| Forest                                 | 6,138      | 20.0%      |
| Built-up (residential, industry, etc.) | 2,762      | 9.0%       |
| Roads and rail infrastructure          | 3,553      | 11.6%      |
| Sport and leisure facilities (others)  | 15         | 0.05%      |
| Golf courses                           | 45         | 0.15%      |
| Football pitches                       | 30         | 0.10%      |
| Ground-mounted PV                      | 13         | 0.04%      |
| Wind turbines (footprint)              | 14         | 0.05%      |
| Paths/tracks                           | 250        | 0.8%       |
| Water bodies (inland)                  | 143        | 0.47%      |
| Other natural/semi-natural/rest        | 4,271      | 13.9%      |

### Total Area: **30,689 km²**

Previous attempts to solve this problem are available in the current folder. However, none has been successfull. An in-depth analysis yields the following strategy, which can be followed and adapted if neeeded:

# Belgium Boundary Data

A highly detailed Belgian boundary shapefile can be obtained from official sources. For example, Statistics Belgium (Statbel) provides open‐data administrative boundaries down to *“statistical sectors”* (\~20 000 polygons).  These come in an equal-area Lambert CRS (e.g. EPSG:31370) which preserves area.  (Other options include Natural Earth or GADM country/subdivision layers if only coarser boundaries are needed.)
This data is downloaded and unzipped in the folder sh_statbel_statistical_sectors_31370_20240101.geojson

# Python Geospatial Libraries

A Python GIS stack should include **GeoPandas** and **Shapely** (with its PyGEOS-backed geometry engine) for handling spatial data. GeoPandas extends pandas with a `GeoDataFrame` that can read/write shapefiles and perform spatial joins and projections.  Shapely provides planar geometry operations (union, intersection, split, etc.) via the GEOS library.  For large datasets, a spatial index (R-tree) can speed up neighbor queries; for example Shapely’s `STRtree` or the `rtree` library can quickly find touching or nearest polygons.  For Voronoi-based methods one can use SciPy’s `scipy.spatial.Voronoi` (or the **geovoronoi** package) and then clip cells to the Belgium boundary using Shapely.  Common numeric and plotting libraries (NumPy, Matplotlib) or even **networkx** (for graph-based approaches) will also be useful.

# Partitioning Algorithm Strategies

Several approaches can create contiguous-area regions of specified size:

* **Region-growing (graph-based)**: Treat each small polygon (e.g. stat. sector) as a node in a planar graph. Build adjacency via spatial join (e.g. GeoDataFrame.sjoin with `predicate='touches'`). Then repeatedly *grow* each region by adding adjacent polygons until the sum of areas reaches the target for that category. This is essentially a breadth-first/greedy expansion ensuring contiguity. It may require backtracking or swapping small pieces at the end to respect the ±3% area tolerance.

* **Spanning-tree partitioning (SKATER-style)**: Compute a minimum spanning tree of the adjacency graph (weight edges arbitrarily). Then partition the tree by cutting edges to form contiguous clusters. For instance, the SKATER algorithm iteratively removes edges to produce regions. Here one could treat each land-use region as a “cluster” and cut the tree to meet area sums, refining by switching branches to adjust areas. These graph-partition methods guarantee contiguity (each cluster is a connected tree) and have been used in spatial regionalization problems.

* **Voronoi or centroidal tessellation**: Seed the territory with one point per category and compute a (possibly weighted) Voronoi diagram clipped to Belgium. Adjust seed positions (Lloyd’s algorithm / centroidal Voronoi) so that cell areas approach the target values. Libraries like SciPy (with Shapely clipping) or **geovoronoi** can generate and adjust Voronoi regions. This tends to produce nicely shaped regions, but exact area matching requires iterative weight adjustment or cell-splitting.

* **Optimization-based**: Formulate an integer program (or use heuristics like AZP/Max-P) where small units are assigned to categories with contiguity constraints. This can be solved by MIP solvers or iterative heuristics, but is computationally complex for many polygons. (It is more common in political redistricting than simple scripting.)

In practice a **region-growing/BFS approach** is simplest to implement for fixed area goals: pick initial seeds (e.g. one per region, possibly near centroids of target areas) and grow each into a contiguous patch until its area hits the target. This can be done sequentially for each category (largest first) or iteratively adjusting all.  The graph-based SKATER/REDCAP style methods provide a useful theoretical framework for contiguous clustering but may be overkill for a one-off script.

# Ensuring Area Accuracy

To meet the ±3% area tolerance, first **project to an equal-area CRS** (e.g. EPSG:3035 or Belgium’s EPSG:31370). Compute each polygon’s area (Shapely’s `.area` gives projected-area in m², convert to km²). During region-growing, keep a running sum of areas. If adding a polygon would exceed the target by more than 3%, one can skip it or try a different neighbor. After initial clustering, check each region’s area against its target; any region outside tolerance can be corrected by swapping small border polygons or (if necessary) splitting a polygon along a line to fine-tune the area.  In code, one might use Shapely’s `split()` or buffering to trim shapes. The ±3% slack allows some flexibility, so usually integer numbers of whole sub-polygons suffice without exact fractional splits.

# Visualization (PNG Output)

Once clusters are finalized, load them into a GeoDataFrame with a “category” column and use GeoPandas/matplotlib to plot. For example:

```python
ax = regions_gdf.plot(column='category', categorical=True, legend=True, figsize=(8,6))
ax.set_title("Belgium land-use regions by category")
plt.savefig("belgium_regions.png")
```

This produces a colored map saved as PNG. GeoPandas abstracts the geometry plotting, and one can customize colors or use contextily to add background tiles if desired.  Matplotlib handles the figure export to PNG.

# Recommended Workflow

A possible script outline is:

1. **Data acquisition:** Download Belgian boundary/units (e.g. Statbel statistical sectors shapefile) via GeoPandas (`gpd.read_file`) or similar. Reproject to equal-area (e.g. `to_crs(epsg=3035)`).
2. **Prepare units:** (Optional) dissolve any irrelevant layers and compute centroids/areas of each small polygon. Build adjacency: e.g. use `sjoin(…, predicate='touches')` or Shapely’s `STRtree`.
3. **Region-growing loop:** For each land-use category (with target area A), pick a start polygon (seed) and repeatedly add the neighboring polygon that keeps the growing region contiguous until the sum of areas ≈ A. Remove assigned polygons from the pool and proceed to the next category. Use a ±3% buffer when checking the sum.
4. **Refinement:** Compute actual areas of all regions and compare to targets. If any are out of tolerance, try swapping border polygons between adjacent regions or splitting a leftover polygon.
5. **Export and plot:** Assemble the final regions into a GeoDataFrame with a category label. Call `plot()` to color them and `plt.savefig()` to produce the PNG image.

This approach leverages **GeoPandas** for I/O and plotting, **Shapely** (and PyGEOS) for geometry ops and area calculation, and spatial indexing to keep the region-growing efficient. By working in an equal-area CRS and checking the cumulative area at each step, the script can ensure each category region meets its area target within ±3%.

**Sources:** Detailed Belgian shapefiles (e.g. Statbel statistical sectors) are publicly available. GeoPandas and Shapely are the standard Python GIS libraries for handling shapefiles and planar geometry. Spatial clustering methods (e.g. graph‐partitioning like SKATER or Voronoi tessellations) inform the algorithmic approach. Reprojecting to Lambert or LAEA ensures area accuracy. The final map can be produced via GeoPandas’ plotting functions into a PNG.
