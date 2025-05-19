import geopandas as gpd
from shapely.geometry import Polygon # Might need more later
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd # Not directly used, geopandas handles its pd dependency
import os

# Define an equal-area CRS
TARGET_CRS = "EPSG:3035" # LAEA Europe

# Land use categories and their target areas in km²
# Based on strategy.md Flattened Land Use Table
LAND_USE_CATEGORIES = [
    {"name": "Agricultural land (others)", "area_km2": 12803, "color": "#FFFFE0"}, # LightYellow
    {"name": "Energy crops", "area_km2": 700, "color": "#FFD700"}, # Gold
    {"name": "Forest", "area_km2": 6138, "color": "#228B22"}, # ForestGreen
    {"name": "Built-up (residential, industry, etc.)", "area_km2": 2762, "color": "#A9A9A9"}, # DarkGray
    {"name": "Roads and rail infrastructure", "area_km2": 3553, "color": "#696969"}, # DimGray
    {"name": "Sport and leisure facilities (others)", "area_km2": 15, "color": "#87CEFA"}, # LightSkyBlue
    {"name": "Golf courses", "area_km2": 45, "color": "#ADD8E6"}, # LightBlue
    {"name": "Football pitches", "area_km2": 30, "color": "#00BFFF"}, # DeepSkyBlue
    {"name": "Ground-mounted PV", "area_km2": 13, "color": "#0000FF"}, # Blue
    {"name": "Wind turbines (footprint)", "area_km2": 14, "color": "#87CEEB"}, # SkyBlue
    {"name": "Paths/tracks", "area_km2": 250, "color": "#808080"}, # Gray
    {"name": "Water bodies (inland)", "area_km2": 143, "color": "#4682B4"}, # SteelBlue
    {"name": "Other natural/semi-natural/rest", "area_km2": 4271, "color": "#90EE90"}, # LightGreen
]

# Total area to be allocated (sum of above)
TOTAL_ALLOCATED_AREA_KM2 = sum(cat["area_km2"] for cat in LAND_USE_CATEGORIES) # Should be 30737

def load_belgium_boundary(filepath="sh_statbel_statistical_sectors_31370_20240101.geojson/sh_statbel_statistical_sectors_31370_20240101.geojson"):
    """Loads the Belgium boundary GeoJSON, reprojects, and calculates area."""
    print(f"Loading boundary file: {filepath}")
    gdf = gpd.read_file(filepath)
    print(f"Original CRS: {gdf.crs}")
    gdf = gdf.to_crs(TARGET_CRS)
    print(f"Reprojected to CRS: {gdf.crs}")

    # Calculate area in km² (Shapely's .area is in units of the CRS, m² for EPSG:3035)
    gdf['area_km2'] = gdf.area / 1_000_000
    total_area_km2 = gdf['area_km2'].sum()
    print(f"Total area of loaded boundary: {total_area_km2:.2f} km²")
    print(f"Number of features in boundary file: {len(gdf)}")
    if len(gdf) == 1:
        print("Boundary file contains a single polygon (country outline).")
    elif len(gdf) > 1:
        print("Boundary file contains multiple polygons (potential base units).")

    return gdf

def prepare_units(boundary_gdf):
    """
    Prepares the base units for region growing.
    If the boundary_gdf is a single polygon, it raises an error.
    Otherwise, it uses the existing polygons as units and calculates adjacency.
    """
    print("\\nStep 2: Preparing units...")
    
    # Ensure the input is not a single MultiPolygon representing the country outline
    # but rather individual polygons if it's a MultiPolygon type.
    if len(boundary_gdf) == 1 and boundary_gdf.geometry.iloc[0].geom_type == 'MultiPolygon':
        print("Input is a single MultiPolygon feature. Exploding into individual Polygons.")
        units_gdf = boundary_gdf.explode(index_parts=True).reset_index(drop=True)
        print(f"Exploded into {len(units_gdf)} individual polygons.")
    elif len(boundary_gdf) == 1 and boundary_gdf.geometry.iloc[0].geom_type == 'Polygon':
         raise ValueError(
            "The input boundary file contains only one single Polygon feature. "
            "Region growing requires a set of smaller 'unit' polygons. "
            "Please provide a file with statistical sectors or similar units, "
            "or implement gridding."
        )
    elif len(boundary_gdf) > 1 :
        print("Using existing polygons from the input file as base units.")
        units_gdf = boundary_gdf.copy()
    else: # len(boundary_gdf) == 0 or other unexpected cases
        raise ValueError("Input boundary file is empty or has an unsupported structure for unit preparation.")


    # Ensure each unit has an area and a unique ID
    units_gdf['unit_id'] = range(len(units_gdf))
    if 'area_km2' not in units_gdf.columns: # Should be there from load_belgium_boundary
        units_gdf['area_km2'] = units_gdf.geometry.area / 1_000_000
    
    # Remove very small units that might cause issues (optional, but good for robustness)
    initial_unit_count = len(units_gdf)
    min_area_threshold_km2 = 0.001 # e.g., 0.001 km^2 = 1000 m^2
    units_gdf = units_gdf[units_gdf['area_km2'] > min_area_threshold_km2]
    if len(units_gdf) < initial_unit_count:
        print(f"Removed {initial_unit_count - len(units_gdf)} units smaller than {min_area_threshold_km2} km².")
    
    if units_gdf.empty:
        raise ValueError("No valid units remained after filtering small polygons. Cannot proceed.")
        
    units_gdf = units_gdf.reset_index(drop=True) # Re-index after potential filtering
    units_gdf['unit_id'] = range(len(units_gdf)) # Re-assign unit_id


    print(f"Prepared {len(units_gdf)} units for adjacency calculation.")
    units_gdf['centroid'] = units_gdf.geometry.centroid # May be useful for seeding later

    # Build adjacency (spatial index for speed)
    print("Building spatial index for adjacency calculations...")
    spatial_index = units_gdf.sindex

    print("Calculating adjacency list (this might take a while for many units)...")
    adjacencies = {}
    for index, unit in units_gdf.iterrows():
        if index % (len(units_gdf)//10 if len(units_gdf) > 20 else 1) == 0 and index > 0: # Progress update
            print(f"  Processed adjacency for {index}/{len(units_gdf)} units...")
        
        # Get candidates using spatial index (intersection with bounds)
        possible_matches_indices = list(spatial_index.intersection(unit.geometry.bounds))
        
        # Filter out self-index if present
        try:
            possible_matches_indices.remove(index)
        except ValueError:
            pass # Index not in list, no problem
            
        possible_matches = units_gdf.iloc[possible_matches_indices]
        
        # Filter to actual touching polygons
        # Using `unit.geometry.buffer(0).touches` can sometimes be more robust for slight imperfections
        precise_matches = possible_matches[possible_matches.geometry.touches(unit.geometry)]
        adjacencies[unit.unit_id] = list(precise_matches.unit_id)
    
    units_gdf['neighbors'] = units_gdf['unit_id'].map(adjacencies)
    print("Adjacency calculation complete.")
    
    # Check for units with no neighbors (islands) - this can be problematic for region growing
    isolated_units = units_gdf[units_gdf['neighbors'].apply(len) == 0]
    if not isolated_units.empty:
        print(f"WARNING: Found {len(isolated_units)} isolated units (no neighbors). These might not be assignable.")
        # Consider how to handle these: assign to closest region post-hoc, or merge into a specific category?
    
    return units_gdf


def initial_region_growing(units_gdf, categories):
    """
    Implements the region-growing algorithm with special handling for the largest category.
    All categories except the largest one are grown using the standard region growing approach.
    The remaining unassigned units are then allocated to the largest category.
    """
    print("\\nStep 3: Initial Region Growing Algorithm...")
    
    # Make a copy of the units_gdf to track assignments
    available_units = units_gdf.copy()
    available_units['assigned_category'] = None
    available_units['region_id'] = -1 

    all_regions_data = [] 
    AREA_TOLERANCE_PCT = 0.03 
    region_id_counter = 0

    # Identify the category with the largest target area
    largest_category = max(categories, key=lambda c: c['area_km2'])
    largest_category_name = largest_category['name']
    largest_category_area = largest_category['area_km2']
    largest_category_color = largest_category['color']
    
    print(f"Identified {largest_category_name} as the largest category with target area {largest_category_area:.2f} km².")
    print(f"This category will be processed last by assigning remaining units after other categories are grown.")
    
    # Filter out the largest category and sort remaining by area (smallest first)
    other_categories = [c for c in categories if c['name'] != largest_category_name]
    sorted_other_categories = sorted(other_categories, key=lambda c: c['area_km2'], reverse=False)
    
    # Process all categories except the largest one
    for category_info in sorted_other_categories:
        category_name = category_info['name']
        target_area_km2 = category_info['area_km2']
        category_color = category_info['color']
        
        print(f"\\nProcessing category: {category_name}, Target Area: {target_area_km2:.2f} km²")

        current_region_unit_indices = [] # Store indices from `available_units`
        current_region_area_km2 = 0.0
        
        unassigned_units_for_cat = available_units[available_units['assigned_category'].isnull()]
        if unassigned_units_for_cat.empty:
            print(f"  WARNING: No unassigned units left to form region for {category_name}.")
            continue

        # Seed selection: Largest available unassigned unit
        # Ensure it's not an isolated unit if possible, unless no other choice
        potential_seeds = unassigned_units_for_cat.copy()
        potential_seeds['num_unassigned_neighbors'] = potential_seeds['neighbors'].apply(
            lambda neigh_ids: sum(1 for nid in neigh_ids if available_units.loc[available_units['unit_id'] == nid, 'assigned_category'].isnull().iloc[0])
        )
        potential_seeds = potential_seeds.sort_values(by=['num_unassigned_neighbors', 'area_km2'], ascending=[False, False])
        
        if potential_seeds.empty: # Should not happen if unassigned_units_for_cat was not empty
             print(f"  WARNING: No potential seeds found for {category_name} (unexpected).")
             continue
        
        seed_unit_original_index = potential_seeds.index[0] # This is an index for `available_units`

        current_region_unit_indices.append(seed_unit_original_index)
        seed_area = available_units.loc[seed_unit_original_index, 'area_km2']
        current_region_area_km2 += seed_area
        available_units.loc[seed_unit_original_index, 'assigned_category'] = category_name
        available_units.loc[seed_unit_original_index, 'region_id'] = region_id_counter
        
        print(f"  Seed unit ID {available_units.loc[seed_unit_original_index, 'unit_id']} (Area: {seed_area:.2f} km²) for {category_name}")

        min_target = target_area_km2 * (1 - AREA_TOLERANCE_PCT)
        max_target = target_area_km2 * (1 + AREA_TOLERANCE_PCT)

        # Iteratively add neighbors
        while True: # Loop will be broken internally
            if current_region_area_km2 >= min_target: # Stop if goal is met or exceeded within tolerance
                if current_region_area_km2 <= max_target:
                    print(f"  Region for {category_name} reached target area ({current_region_area_km2:.2f} km²).")
                    break 
                # If slightly over max_target, still break (may need refinement later)
                # This simple break might leave it slightly over. Refinement could trim.
                # For now, if it's >= min_target, we consider it mostly done.
                # If it significantly overshot on the last addition, that's a limitation.
                print(f"  Region for {category_name} is {current_region_area_km2:.2f} km² (Target: {target_area_km2:.2f}, Max: {max_target:.2f}). Stopping.")
                break


            # Gather frontier: unassigned neighbors of the current region's units
            frontier_unit_ids_set = set()
            for unit_idx_in_available in current_region_unit_indices:
                # Get the 'unit_id' to look up neighbors in the original structure (units_gdf or available_units)
                # Assuming 'neighbors' column in available_units holds the unit_ids of neighbors
                neighboring_unit_ids = available_units.loc[unit_idx_in_available, 'neighbors']
                if isinstance(neighboring_unit_ids, list):
                    frontier_unit_ids_set.update(neighboring_unit_ids)
            
            # Get actual candidate units from `available_units` using their indices
            candidate_unit_indices = available_units[
                (available_units['unit_id'].isin(list(frontier_unit_ids_set))) &
                (available_units['assigned_category'].isnull())
            ].index.tolist()

            if not candidate_unit_indices:
                print(f"  WARNING: No available unassigned neighbors to expand {category_name}. Current area {current_region_area_km2:.2f} km² (Target: {min_target:.2f}-{max_target:.2f} km²).")
                break 

            # Select best candidate to add
            best_candidate_to_add_idx = -1
            
            # Strategy: Prioritize candidates that don't make the region exceed max_target.
            # Among those, pick the one that results in an area closest to target_area_km2 OR largest.
            # For simplicity: pick largest that doesn't exceed max_target.
            # If all exceed max_target, but we are below min_target, pick smallest to minimize overshoot.
            
            possible_additions = []
            for cand_original_idx in candidate_unit_indices:
                cand_area = available_units.loc[cand_original_idx, 'area_km2']
                possible_additions.append({
                    'index': cand_original_idx, # Index in available_units
                    'area': cand_area,
                    'new_total_area': current_region_area_km2 + cand_area
                })

            # Candidates that don't overshoot max_target
            suitable_additions = [p for p in possible_additions if p['new_total_area'] <= max_target]

            if suitable_additions:
                # Pick the one that gets us closest to target_area_km2 (or largest of these)
                suitable_additions.sort(key=lambda x: abs(x['new_total_area'] - target_area_km2)) # Closest to target
                # suitable_additions.sort(key=lambda x: x['area'], reverse=True) # Or largest area
                best_candidate_to_add_idx = suitable_additions[0]['index']
            elif current_region_area_km2 < min_target: 
                # No suitable additions found, but we are still under min_target. Must add something if possible.
                # Pick the one that causes the smallest overshoot (i.e., smallest area)
                possible_additions.sort(key=lambda x: x['area']) # Smallest area first
                if possible_additions:
                    best_candidate_to_add_idx = possible_additions[0]['index']
                    print(f"    Forced to pick unit (ID {available_units.loc[best_candidate_to_add_idx, 'unit_id']}, Area {available_units.loc[best_candidate_to_add_idx, 'area_km2']:.2f}) for {category_name} as current area {current_region_area_km2:.2f} is below min_target {min_target:.2f}")
            
            if best_candidate_to_add_idx != -1:
                unit_to_add_area = available_units.loc[best_candidate_to_add_idx, 'area_km2']
                unit_to_add_id = available_units.loc[best_candidate_to_add_idx, 'unit_id']
                # print(f"    Adding unit {unit_to_add_id} (Area: {unit_to_add_area:.2f} km²) to {category_name}. New area: {current_region_area_km2 + unit_to_add_area:.2f} km²")
                
                current_region_unit_indices.append(best_candidate_to_add_idx)
                current_region_area_km2 += unit_to_add_area
                available_units.loc[best_candidate_to_add_idx, 'assigned_category'] = category_name
                available_units.loc[best_candidate_to_add_idx, 'region_id'] = region_id_counter
            else:
                # No suitable candidate found and not forced to pick one, or frontier exhausted.
                print(f"  Stopping growth for {category_name} at {current_region_area_km2:.2f} km² (Target: {min_target:.2f}-{max_target:.2f} km²). No suitable units to add.")
                break # Exit while loop for this category
        
        # Region for this category is complete (or growth stopped)
        print(f"  Completed growth for {category_name}. Final Area: {current_region_area_km2:.2f} km² (Target: {target_area_km2:.2f} km²).")
        
        if current_region_unit_indices:
            region_parts_gdf = available_units.loc[current_region_unit_indices]
            if not region_parts_gdf.empty:
                # It's crucial to handle cases where region_parts_gdf might be invalid for union_all
                try:
                    # Ensure geometries are valid before union
                    # region_parts_gdf.geometry = region_parts_gdf.geometry.buffer(0) # Simple cleaning
                    region_polygon = region_parts_gdf.geometry.union_all()
                    all_regions_data.append({
                        'geometry': region_polygon, 
                        'category': category_name,
                        'area_km2_summed': current_region_area_km2, # Area from summing units
                        'target_area_km2': target_area_km2,
                        'color': category_color,
                        'region_id': region_id_counter
                    })
                except Exception as e:
                    print(f"  ERROR: Could not form final polygon for {category_name} (units: {len(region_parts_gdf)}). Error: {e}")
                    print(f"  Skipping region for {category_name}. Units will remain assigned but not part of final GDF.")

        region_id_counter += 1

    # Now handle the largest category by assigning all remaining unassigned units
    unassigned_final = available_units[available_units['assigned_category'].isnull()]
    
    if not unassigned_final.empty:
        unassigned_area_sum = unassigned_final['area_km2'].sum()
        print(f"\\nAssigning remaining unassigned units to the largest category ({largest_category_name})...")
        print(f"  {len(unassigned_final)} units ({unassigned_area_sum:.2f} km²) available to assign.")
        print(f"  Target for {largest_category_name}: {largest_category_area:.2f} km²")
        
        # All units being assigned to largest category
        largest_cat_units = unassigned_final.copy()
        largest_cat_units_indices = largest_cat_units.index.tolist()
        
        # Assign all these to the largest category
        for idx in largest_cat_units_indices:
            available_units.loc[idx, 'assigned_category'] = largest_category_name
            available_units.loc[idx, 'region_id'] = region_id_counter
        
        # Create the geometry for the largest category
        try:
            largest_cat_polygon = largest_cat_units.geometry.union_all()
            all_regions_data.append({
                'geometry': largest_cat_polygon, 
                'category': largest_category_name,
                'area_km2_summed': unassigned_area_sum, # Area from summing units
                'target_area_km2': largest_category_area,
                'color': largest_category_color,
                'region_id': region_id_counter
            })
            print(f"  Assigned area for {largest_category_name}: {unassigned_area_sum:.2f} km² (Target: {largest_category_area:.2f} km²)")
            region_id_counter += 1
        except Exception as e:
            print(f"  ERROR: Could not form final polygon for {largest_category_name} (units: {len(largest_cat_units)}). Error: {e}")
            print(f"  Largest category ({largest_category_name}) may not be correctly represented in the final output.")
    else:
        print(f"\\nWARNING: No units left to assign to the largest category ({largest_category_name})!")
        print(f"This likely means the other categories consumed all available units.")
        # Add an empty placeholder for the largest category
        all_regions_data.append({
            'geometry': None,  # This will likely cause issues later, but we need to represent the category
            'category': largest_category_name,
            'area_km2_summed': 0,
            'target_area_km2': largest_category_area,
            'color': largest_category_color,
            'region_id': region_id_counter
        })
    
    # Check if there are ANY remaining unassigned units (should be none)
    unassigned_check = available_units[available_units['assigned_category'].isnull()]
    if not unassigned_check.empty:
        print(f"\\nWARNING: {len(unassigned_check)} units ({unassigned_check['area_km2'].sum():.2f} km²) somehow remained unassigned after all categories processed.")
        print("This is unexpected - all units should now be assigned!")

    # Create the final GeoDataFrame with all regions
    if not all_regions_data:
        print("WARNING: No regions were successfully created.")
        return gpd.GeoDataFrame(), available_units

    final_regions_gdf = gpd.GeoDataFrame(all_regions_data, crs=TARGET_CRS)
    # Recalculate geometric area for accuracy check
    final_regions_gdf['area_km2_geom'] = final_regions_gdf.geometry.area / 1_000_000
    
    return final_regions_gdf, available_units

def smart_swapping_refinement(regions_gdf, units_with_assignments_gdf, all_units_gdf, categories):
    """
    Refines the regions by swapping units between neighboring regions to improve area accuracy.
    Uses distance from region center of mass to prioritize swaps that make regions more compact.
    
    This is a placeholder for future implementation of the smart swapping algorithm.
    """
    print("\\nStep 4: Smart Swapping Refinement...")
    print("NOTE: This is currently a placeholder. No region refinement will occur.")
    print("A future implementation will use center of mass calculations and unit swapping to:")
    print("  1. Improve area accuracy for regions not meeting their targets")
    print("  2. Make regions more compact by prioritizing units closer to region centers")
    print("  3. Reduce region fragmentation")
    
    # For now, just return the regions as they are
    return regions_gdf

def refine_regions(regions_gdf, units_with_assignments_gdf, categories):
    """Placeholder for refinement step."""
    print("\\nStep 4: Refinement (Placeholder)...")
    # This function would ideally:
    # 1. Check area tolerances for each region in regions_gdf.
    # 2. For regions out of tolerance:
    #    a. Identify border units using `units_with_assignments_gdf`.
    #    b. Attempt to swap small border units with adjacent regions to meet targets.
    #    c. This is complex, involving finding neighbors across different `region_id`s.
    print("Current refinement step is a placeholder and does not modify regions.")
    return regions_gdf 

def export_and_plot(regions_gdf, filename_stem="belgium_land_use"):
    """Exports regions to GeoJSON and plots them to PNG."""
    print("\\nStep 5: Export and Plot...")

    if regions_gdf.empty or 'geometry' not in regions_gdf.columns:
        print("No valid regions with geometry to export or plot.")
        return

    # Ensure 'category' and 'color' columns exist for plotting
    if 'category' not in regions_gdf.columns:
        print("WARNING: 'category' column missing in regions_gdf. Plotting may be affected.")
        regions_gdf['category'] = 'Unknown' # Fallback
    if 'color' not in regions_gdf.columns:
        print("WARNING: 'color' column missing in regions_gdf. Plotting may use default colors.")
        # Create a fallback color mapping if needed
        temp_color_map = {cat_info['name']: cat_info['color'] for cat_info in LAND_USE_CATEGORIES}
        temp_color_map['Unassigned'] = '#D3D3D3'
        temp_color_map['Unknown'] = '#000000'
        regions_gdf['color'] = regions_gdf['category'].map(temp_color_map).fillna('#000000')


    geojson_filename = f"{filename_stem}_regions.geojson"
    try:
        # Make a copy for export to avoid altering the GDF passed to check_function
        export_gdf = regions_gdf.copy()
        # GeoJSON does not like all data types, e.g. if area_km2_summed is there but area_km2_geom is preferred
        if 'area_km2_summed' in export_gdf.columns and 'area_km2_geom' in export_gdf.columns:
             del export_gdf['area_km2_summed'] # Keep the geom one as primary
        export_gdf.to_file(geojson_filename, driver="GeoJSON")
        print(f"Regions exported to {geojson_filename}")
    except Exception as e:
        print(f"Error exporting to GeoJSON: {e}")

    print(f"Plotting regions to {filename_stem}_regions.png...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10)) # Increased fig size for legend
    
    unique_categories = regions_gdf['category'].unique()
    
    for category_name in unique_categories:
        data_to_plot = regions_gdf[regions_gdf['category'] == category_name]
        # Get color from the first row of this category's data, or default
        color_val = data_to_plot['color'].iloc[0] if not data_to_plot.empty and 'color' in data_to_plot.columns else '#000000'
        data_to_plot.plot(ax=ax, color=color_val, label=category_name, edgecolor='darkgrey', linewidth=0.3)

    ax.set_title("Belgium Land Use Regions by Category", fontsize=16)
    ax.set_xlabel("Easting (m, EPSG:3035)", fontsize=12)
    ax.set_ylabel("Northing (m, EPSG:3035)", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    # ax.set_aspect('equal', adjustable='box') # This can make plots very wide/tall if extent is not square

    import matplotlib.patches as mpatches
    legend_patches = []
    # Ensure consistent legend order, perhaps matching LAND_USE_CATEGORIES + Unassigned
    ordered_legend_cats = [cat['name'] for cat in LAND_USE_CATEGORIES]
    if "Unassigned" in unique_categories and "Unassigned" not in ordered_legend_cats:
        ordered_legend_cats.append("Unassigned")
    if "Unknown" in unique_categories and "Unknown" not in ordered_legend_cats:
         ordered_legend_cats.append("Unknown")


    for cat_name in ordered_legend_cats:
        if cat_name in unique_categories:
            # Get color from the original GDF for this category
            cat_data = regions_gdf[regions_gdf['category'] == cat_name]
            if not cat_data.empty:
                 color_val = cat_data['color'].iloc[0]
                 legend_patches.append(mpatches.Patch(color=color_val, label=cat_name))
            elif cat_name == "Unassigned": # Special handle if it wasn't plotted but should be in legend
                 legend_patches.append(mpatches.Patch(color="#D3D3D3", label="Unassigned"))


    ax.legend(handles=legend_patches, title="Land Use Category", loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, title_fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust layout for external legend

    png_filename = f"{filename_stem}_regions.png"
    try:
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        print(f"Map saved to {png_filename}")
    except Exception as e:
        print(f"Error saving PNG: {e}")
    plt.close(fig)


def check_function(final_regions_gdf, categories_definition, original_total_boundary_area_km2):
    """Verifies region presence, area tolerance, and total coverage."""
    print("\\nVerification Check:")
    if final_regions_gdf.empty or 'geometry' not in final_regions_gdf.columns:
        print("No valid regions generated to check.")
        return False

    all_ok = True
    AREA_TOLERANCE_PCT = 0.03
    total_script_allocated_geom_area = final_regions_gdf['area_km2_geom'].sum()
    
    category_map_from_def = {cat['name']: cat for cat in categories_definition}

    print(f"{'Category':<40} {'Target (km²)':<15} {'Actual (km²)':<15} {'% Diff':<10} {'Tolerance Met?':<15}")
    print("-" * 100)

    for idx, region in final_regions_gdf.iterrows():
        category_name = region['category']
        actual_geom_area = region['area_km2_geom']
        
        if category_name == "Unassigned":
            if actual_geom_area > 0.01: # Tolerance for small unassigned slivers
                print(f"{category_name:<40} {'N/A':<15} {actual_geom_area:<15.2f} {'N/A':<10} {'WARNING (Unassigned)':<15}")
            continue

        category_def = category_map_from_def.get(category_name)
        if not category_def:
            print(f"ERROR: Category '{category_name}' found in regions but not in definitions.")
            all_ok = False
            continue
        
        target_area = category_def['area_km2']
        min_target = target_area * (1 - AREA_TOLERANCE_PCT)
        max_target = target_area * (1 + AREA_TOLERANCE_PCT)
        
        percent_diff = ((actual_geom_area - target_area) / target_area) * 100 if target_area else 0
        status = "YES"
        if not (min_target <= actual_geom_area <= max_target):
            status = "NO"
            all_ok = False
        
        print(f"{category_name:<40} {target_area:<15.2f} {actual_geom_area:<15.2f} {percent_diff:<9.2f}% {status:<15}")

    defined_category_names = set(cat['name'] for cat in categories_definition)
    output_category_names = set(final_regions_gdf[final_regions_gdf['category'] != 'Unassigned']['category'].unique())
    
    missing_categories = defined_category_names - output_category_names
    if missing_categories:
        print(f"\\nERROR: Defined categories missing from output: {missing_categories}")
        all_ok = False
    
    extra_categories = output_category_names - defined_category_names
    if extra_categories:
        print(f"\\nWARNING: Output has extra categories (not in definitions): {extra_categories}")

    print("-" * 100)
    print(f"Sum of Target Areas (from LAND_USE_CATEGORIES): {TOTAL_ALLOCATED_AREA_KM2:.2f} km²")
    print(f"Sum of Actual Generated Region Areas (geometric): {total_script_allocated_geom_area:.2f} km²")
    print(f"Total Area of Original Boundary Input: {original_total_boundary_area_km2:.2f} km²")

    # Check if total generated area matches the sum of targets within a small global tolerance
    if abs(total_script_allocated_geom_area - TOTAL_ALLOCATED_AREA_KM2) > TOTAL_ALLOCATED_AREA_KM2 * 0.01: # 1% overall diff
        print(f"WARNING: Total generated area ({total_script_allocated_geom_area:.2f} km²) differs >1% from sum of targets ({TOTAL_ALLOCATED_AREA_KM2:.2f} km²).")
        # all_ok = False # Could make this a failure condition

    # Check if total generated area is close to the original boundary area (minus unassigned)
    # This assumes the sum of targets (TOTAL_ALLOCATED_AREA_KM2) is what should fill the boundary.
    # A small discrepancy is expected due to "Paths/tracks" not being part of TOTAL_ALLOCATED_AREA_KM2.
    # Difference should be around the area of "Paths/tracks" (250km2) if TOTAL_ALLOCATED_AREA_KM2 is less than original_total_boundary_area_km2
    
    discrepancy_with_boundary = original_total_boundary_area_km2 - total_script_allocated_geom_area
    print(f"Discrepancy (Boundary Total - Generated Total): {discrepancy_with_boundary:.2f} km²")
    # This discrepancy might represent unallocated parts of the original boundary if TOTAL_ALLOCATED_AREA_KM2 was less than it.

    if all_ok:
        print("\\nVerification PASSED (based on checks implemented).")
    else:
        print("\\nVerification FAILED.")
    return all_ok

def main():
    """Main function to orchestrate the land division process."""
    import os
    
    # File paths for intermediate results
    INITIAL_REGIONS_PATH = "initial_regions.geojson"
    ASSIGNED_UNITS_PATH = "assigned_units_after_growth.geojson"
    FINAL_RESULTS_PATH = "belgium_land_use_custom_regions"

    print("Starting Belgium Land Use Divider Script...")
    print(f"Target total allocated area from definitions: {TOTAL_ALLOCATED_AREA_KM2:.2f} km²")
    original_boundary_total_area_km2 = 0.0

    # Phase 1: Prepare Units - always run this step
    try:
        belgium_gdf = load_belgium_boundary(filepath="sh_statbel_statistical_sectors_31370_20240101.geojson/sh_statbel_statistical_sectors_31370_20240101.geojson")
        original_boundary_total_area_km2 = belgium_gdf['area_km2'].sum() # Store for final check
        units_gdf = prepare_units(belgium_gdf)
    except FileNotFoundError:
        print("Error: 'sh_statbel_statistical_sectors_31370_20240101.geojson' not found. Please ensure it's in the script's directory.")
        return
    except ValueError as ve:
        print(f"CRITICAL ERROR in preparing units: {ve}")
        print("This typically means 'sh_statbel_statistical_sectors_31370_20240101.geojson' is a single outline or unsuitable.")
        print("Provide a GeoJSON/Shapefile with smaller administrative units (e.g., statistical sectors).")
        return
    except Exception as e:
        print(f"An unexpected CRITICAL error occurred during unit preparation: {e}")
        import traceback
        traceback.print_exc()
        return
            
    if units_gdf.empty or 'neighbors' not in units_gdf.columns:
        print("CRITICAL: Unit preparation failed or did not produce required 'neighbors' column. Exiting.")
        return

    # Phase 2: Check for existing initial regions and assigned units
    initial_regions_gdf = None
    units_with_assignments_gdf = None
    
    if os.path.exists(INITIAL_REGIONS_PATH) and os.path.exists(ASSIGNED_UNITS_PATH):
        try:
            print(f"Found initial regions file: {INITIAL_REGIONS_PATH}")
            print(f"Found assigned units file: {ASSIGNED_UNITS_PATH}")
            print("Loading initial regions and assigned units...")
            initial_regions_gdf = gpd.read_file(INITIAL_REGIONS_PATH)
            print(f"Loaded {len(initial_regions_gdf)} regions.")
            
            # For now, we'll skip loading assigned units since we're not actively using them
            # in smart_swapping_refinement yet. When we implement that function, we'll need
            # to revisit this decision.
        except Exception as e:
            print(f"Error loading initial regions: {e}")
            initial_regions_gdf = None
    
    # If we couldn't load initial regions, generate them
    if initial_regions_gdf is None:
        try:
            initial_regions_gdf, units_with_assignments_gdf = initial_region_growing(units_gdf, LAND_USE_CATEGORIES)
            
            # Save initial regions (we'll skip saving assigned units for now)
            print(f"Saving initial regions to {INITIAL_REGIONS_PATH} for future runs...")
            initial_regions_gdf.to_file(INITIAL_REGIONS_PATH, driver="GeoJSON")
            print("Initial regions saved.")
        except Exception as e:
            print(f"CRITICAL Error during initial region growing: {e}")
            import traceback
            traceback.print_exc()
            return

    if initial_regions_gdf.empty:
        print("CRITICAL: Region growing did not produce any regions. Exiting.")
        return
    
    # Phase 3: Smart Swapping Refinement (currently a placeholder)
    final_regions_gdf = smart_swapping_refinement(initial_regions_gdf, units_with_assignments_gdf, units_gdf, LAND_USE_CATEGORIES)
    
    # Phase 4: Verification and Export
    export_and_plot(final_regions_gdf, filename_stem=FINAL_RESULTS_PATH)
    check_function(final_regions_gdf, LAND_USE_CATEGORIES, original_boundary_total_area_km2)
    
    print("\\nScript finished.")

if __name__ == "__main__":
    main() 