import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import signal
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Global timeout settings
TIMEOUT_SECONDS = 120  # 2 minutes
START_TIME = None

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Algorithm timed out after 2 minutes")

def check_timeout(phase_name=""):
    """Check if we're approaching the timeout and should terminate early"""
    global START_TIME
    if START_TIME is None:
        return False
    
    elapsed = time.time() - START_TIME
    if elapsed > TIMEOUT_SECONDS - 10:  # Stop 10 seconds before timeout
        print(f"  ⏰ Approaching timeout ({elapsed:.1f}s), terminating {phase_name} early")
        return True
    return False

# Define an equal-area CRS
TARGET_CRS = "EPSG:3035" # LAEA Europe

# Land use categories and their target areas in km²
# Ordered by target area (largest first for better region growing)
LAND_USE_CATEGORIES = [
    {"name": "Agricultural land (others)", "area_km2": 12803, "color": "#FFFFE0"},
    {"name": "Forest", "area_km2": 6138, "color": "#228B22"},
    {"name": "Other natural/semi-natural/rest", "area_km2": 4271, "color": "#90EE90"},
    {"name": "Roads and rail infrastructure", "area_km2": 3553, "color": "#696969"},
    {"name": "Built-up (residential, industry, etc.)", "area_km2": 2762, "color": "#A9A9A9"},
    {"name": "Energy crops", "area_km2": 700, "color": "#FFD700"},
    {"name": "Paths/tracks", "area_km2": 250, "color": "#808080"},
    {"name": "Water bodies (inland)", "area_km2": 143, "color": "#4682B4"},
    {"name": "Golf courses", "area_km2": 45, "color": "#ADD8E6"},
    {"name": "Football pitches", "area_km2": 30, "color": "#00BFFF"},
    {"name": "Sport and leisure facilities (others)", "area_km2": 15, "color": "#87CEFA"},
    {"name": "Wind turbines (footprint)", "area_km2": 14, "color": "#87CEEB"},
    {"name": "Ground-mounted PV", "area_km2": 13, "color": "#0000FF"},
]

# Total area to be allocated
TOTAL_ALLOCATED_AREA_KM2 = sum(cat["area_km2"] for cat in LAND_USE_CATEGORIES)
AREA_TOLERANCE_PCT = 0.03

def build_efficient_adjacency(units_gdf):
    """Use spatial index with optimized neighbor finding and timeout protection"""
    print("🔗 BUILDING EFFICIENT SPATIAL ADJACENCY USING STRtree...")
    print("-" * 60)
    start_time = time.time()
    
    # Build spatial index
    print(f"📊 Creating spatial index for {len(units_gdf)} units...")
    geometries = units_gdf.geometry.tolist()
    tree = STRtree(geometries)
    print(f"✅ Spatial index created in {time.time() - start_time:.2f} seconds")
    
    adjacencies = {}
    unit_id_to_idx = {row['unit_id']: idx for idx, row in units_gdf.iterrows()}
    
    progress_step = len(units_gdf) // 10  # Reduced frequency for speed
    processed = 0
    last_update_time = start_time
    
    print(f"\n🔍 PROCESSING ADJACENCY RELATIONSHIPS:")
    for idx, unit in units_gdf.iterrows():
        # Check timeout every 1000 units
        if processed % 1000 == 0 and check_timeout("adjacency calculation"):
            print(f"  ⏰ Timeout approaching, processed {processed}/{len(units_gdf)} units")
            break
            
        if processed % progress_step == 0 or time.time() - last_update_time > 3:
            progress = processed / len(units_gdf) * 100
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            estimated_remaining = (len(units_gdf) - processed) / rate if rate > 0 else 0
            print(f"  📊 Progress: {progress:>5.1f}% ({processed:>5}/{len(units_gdf)}) - "
                  f"{elapsed:>5.1f}s elapsed - {rate:>5.1f} units/sec - ETA: {estimated_remaining:>5.1f}s")
            last_update_time = time.time()
        
        # Get candidates efficiently using spatial index
        candidate_indices = tree.query(unit.geometry, predicate='intersects')
        
        # Filter to actual neighbors (touching, not just intersecting)
        neighbors = []
        for candidate_idx in candidate_indices:
            if candidate_idx != idx:
                candidate_unit = units_gdf.iloc[candidate_idx]
                if unit.geometry.touches(candidate_unit.geometry):
                    neighbors.append(candidate_unit['unit_id'])
        
        adjacencies[unit['unit_id']] = neighbors
        processed += 1
    
    end_time = time.time()
    neighbor_counts = [len(neighbors) for neighbors in adjacencies.values()]
    avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
    max_neighbors = max(neighbor_counts) if neighbor_counts else 0
    
    print(f"\n✅ ADJACENCY CALCULATION COMPLETED:")
    print(f"   Duration: {end_time - start_time:.2f} seconds")
    print(f"   Processed: {processed}/{len(units_gdf)} units ({processed/len(units_gdf)*100:.1f}%)")
    print(f"   Average neighbors per unit: {avg_neighbors:.2f}")
    print(f"   Maximum neighbors: {max_neighbors}")
    print(f"   Performance: {processed/(end_time - start_time):.1f} units/second")
    
    return adjacencies

def select_optimal_seeds(units_gdf, categories, adjacencies):
    """Select optimal seed units for each category based on connectivity and area distribution"""
    print("🌱 SELECTING OPTIMAL SEEDS FOR REGION GROWING...")
    print("-" * 60)
    start_time = time.time()
    
    # Create efficient lookup for unit centroids
    print(f"📊 Preparing seed selection for {len(categories)} categories...")
    unit_centroids = {row['unit_id']: row.geometry.centroid for _, row in units_gdf.iterrows()}
    total_area = units_gdf['area_km2'].sum()
    
    seeds = {}
    
    for i, category in enumerate(categories, 1):
        category_name = category['name']
        target_area = category['area_km2']
        
        print(f"\n🔍 {i:2}/{len(categories)} Selecting seed for {category_name} (target: {target_area:.0f} km²)...")
        
        # Calculate expected position based on area distribution
        area_fraction = target_area / total_area
        print(f"     Target area fraction: {area_fraction:.3f} ({area_fraction*100:.1f}%)")
        
        # Score all units for this category
        best_score = -float('inf')
        best_unit_id = None
        best_unit_area = 0
        candidates_evaluated = 0
        
        # Sample units for efficiency
        sample_size = min(1000, len(units_gdf))
        sample_units = units_gdf.sample(n=sample_size, random_state=42)
        
        for _, unit in sample_units.iterrows():
            unit_id = unit['unit_id']
            unit_area = unit['area_km2']
            
            # Calculate connectivity score
            neighbor_count = len(adjacencies.get(unit_id, []))
            
            # Calculate position score (prefer central positions for large categories)
            unit_centroid = unit_centroids[unit_id]
            
            # Distance to geographic center of Belgium (approximate)
            belgium_center_x, belgium_center_y = 4.3517, 50.8503  # Brussels approx
            # Convert to projected coordinates if needed (simplified)
            
            # Combined score
            connectivity_score = min(neighbor_count / 10.0, 1.0)  # Normalize to 0-1
            area_score = min(unit_area / 10.0, 1.0)  # Prefer reasonable sized units
            
            # For larger categories, prefer more central and well-connected units
            if target_area > 1000:  # Large categories
                final_score = connectivity_score * 2.0 + area_score
            else:  # Small categories
                final_score = connectivity_score + area_score * 2.0
            
            if final_score > best_score:
                best_score = final_score
                best_unit_id = unit_id
                best_unit_area = unit_area
            
            candidates_evaluated += 1
        
        if best_unit_id is not None:
            seeds[category_name] = best_unit_id
            neighbor_count = len(adjacencies.get(best_unit_id, []))
            completion_pct = (best_unit_area / target_area) * 100
            print(f"     ✅ Selected unit {best_unit_id} (area: {best_unit_area:.3f} km², score: {best_score:.3f})")
            print(f"     📊 {neighbor_count} neighbors, {completion_pct:.3f}% of target, evaluated {candidates_evaluated} candidates")
        else:
            print(f"     ❌ No suitable seed found for {category_name}")
    
    end_time = time.time()
    print(f"\n✅ SEED SELECTION COMPLETED:")
    print(f"   Duration: {end_time - start_time:.2f} seconds")
    print(f"   Seeds selected: {len(seeds)}/{len(categories)} categories")
    
    return seeds

def controlled_contiguous_region_growing(units_gdf, categories, adjacencies, seeds):
    """Grow regions with controlled area limits and timeout protection"""
    print("\n🚀 STARTING CONTROLLED CONTIGUOUS REGION GROWING...")
    print("-" * 60)
    start_time = time.time()
    
    # Initialize tracking structures
    unit_assignments = {}  # unit_id -> category_name
    region_areas = {cat['name']: 0.0 for cat in categories}
    region_targets = {cat['name']: cat['area_km2'] for cat in categories}
    
    # Unit lookup for efficiency
    unit_lookup = {row['unit_id']: row for _, row in units_gdf.iterrows()}
    total_units = len(units_gdf)
    
    print(f"📊 Initialized tracking for {total_units} units across {len(categories)} categories")
    
    # Track frontiers for all categories
    frontiers = {cat['name']: set() for cat in categories}
    
    # Initialize all regions with their seeds
    print(f"\n🌱 INITIALIZING REGIONS WITH SEEDS:")
    for i, category in enumerate(categories, 1):
        category_name = category['name']
        target_area = category['area_km2']
        
        if category_name not in seeds:
            print(f"  {i:2}/{len(categories)} ⚠️  {category_name} - NO SEED FOUND, SKIPPING")
            continue
        
        # Initialize region with seed
        seed_unit_id = seeds[category_name]
        unit_assignments[seed_unit_id] = category_name
        seed_unit = unit_lookup[seed_unit_id]
        region_areas[category_name] = seed_unit['area_km2']
        
        # Initialize frontier with seed's unassigned neighbors
        neighbor_count = 0
        for neighbor_id in adjacencies.get(seed_unit_id, []):
            if neighbor_id not in unit_assignments:
                frontiers[category_name].add(neighbor_id)
                neighbor_count += 1
        
        completion_pct = (seed_unit['area_km2'] / target_area) * 100 if target_area > 0 else 0
        print(f"  {i:2}/{len(categories)} ✅ {category_name:<30} - Seed: {seed_unit_id:>5} ({seed_unit['area_km2']:>6.2f} km²) "
              f"Target: {target_area:>7.0f} km² ({completion_pct:>5.2f}%) - {neighbor_count} neighbors")
    
    print(f"\n📈 CONTROLLED GROWTH PHASE - AREA LIMITS: 105% OF TARGET")
    print("-" * 70)
    
    # Controlled growth phase - grow all regions simultaneously with strict area limits
    iteration = 0
    max_iterations = 10000  # Reduced for timeout protection
    last_progress_time = start_time
    last_status_time = start_time
    
    while any(frontiers.values()) and iteration < max_iterations:
        # Check timeout every 100 iterations
        if iteration % 100 == 0 and check_timeout("region growing"):
            break
            
        iteration += 1
        
        # Progress updates every 500 iterations or every 5 seconds
        if iteration % 500 == 0 or time.time() - last_progress_time > 5:
            assigned_count = len(unit_assignments)
            total_count = len(units_gdf)
            progress = assigned_count / total_count * 100
            elapsed = time.time() - start_time
            rate = assigned_count / elapsed if elapsed > 0 else 0
            print(f"  📊 Iteration {iteration:>5}: {assigned_count:>5}/{total_count} units assigned ({progress:>5.1f}%) "
                  f"- {elapsed:>5.1f}s elapsed - {rate:>5.1f} units/sec")
            last_progress_time = time.time()
        
        # Detailed status every 10 seconds
        if time.time() - last_status_time > 10:
            print(f"  📋 CATEGORY STATUS AT ITERATION {iteration}:")
            for category in categories[:5]:  # Show first 5 categories
                category_name = category['name']
                current_area = region_areas[category_name]
                target_area = region_targets[category_name]
                completion = (current_area / target_area) * 100 if target_area > 0 else 0
                frontier_size = len(frontiers[category_name])
                print(f"     {category_name:<25} {completion:>6.1f}% complete ({current_area:>7.1f}/{target_area:>7.0f} km²) "
                      f"- {frontier_size:>4} frontiers")
            if len(categories) > 5:
                print(f"     ... and {len(categories) - 5} more categories")
            last_status_time = time.time()
        
        # Sort categories by priority - those most under their target first
        active_categories = []
        for category in categories:
            category_name = category['name']
            if not frontiers[category_name]:
                continue
                
            current_area = region_areas[category_name]
            target_area = region_targets[category_name]
            completion_ratio = current_area / target_area if target_area > 0 else 1
            
            # Only consider categories that haven't exceeded their limit
            if completion_ratio < 1.05:  # Allow 5% overshoot max
                priority = (target_area - current_area) / target_area if target_area > 0 else 0
                active_categories.append((priority, category_name, category))
        
        if not active_categories:
            print(f"  🏁 All categories reached their limits at iteration {iteration}")
            break
        
        # Sort by priority (highest deficit first)
        active_categories.sort(reverse=True)
        
        units_added_this_iteration = 0
        
        # Try to add one unit to each active category
        for priority, category_name, category in active_categories:
            current_area = region_areas[category_name]
            target_area = region_targets[category_name]
            max_allowed = target_area * 1.05  # 5% overshoot max
            
            if current_area >= max_allowed:
                continue
            
            # Clean frontier - remove already assigned units
            frontiers[category_name] = {uid for uid in frontiers[category_name] 
                                      if uid not in unit_assignments}
            
            if not frontiers[category_name]:
                continue
            
            # Find best candidate from frontier (OPTIMIZED - limit search)
            best_candidate_id = None
            best_score = -float('inf')
            
            # Sample frontier for performance - much smaller sample for speed
            frontier_sample = list(frontiers[category_name])
            if len(frontier_sample) > 15:  # Reduced from 30 for speed
                np.random.shuffle(frontier_sample)
                frontier_sample = frontier_sample[:15]
            
            for candidate_id in frontier_sample:
                if candidate_id in unit_assignments:
                    continue
                
                candidate_unit = unit_lookup[candidate_id]
                new_area = current_area + candidate_unit['area_km2']
                
                # Skip if this would exceed the strict limit
                if new_area > max_allowed:
                    continue
                
                # Calculate score prioritizing connectivity and area fit
                score = calculate_controlled_candidate_score(
                    candidate_unit, category_name, current_area, target_area,
                    adjacencies, unit_assignments
                )
                
                if score > best_score:
                    best_score = score
                    best_candidate_id = candidate_id
            
            # Add best candidate if found
            if best_candidate_id is not None:
                unit_assignments[best_candidate_id] = category_name
                candidate_unit = unit_lookup[best_candidate_id]
                region_areas[category_name] += candidate_unit['area_km2']
                units_added_this_iteration += 1
                
                # Update frontiers: remove added unit from all frontiers
                for cat_name in frontiers:
                    frontiers[cat_name].discard(best_candidate_id)
                
                # Add new neighbors to this category's frontier
                for neighbor_id in adjacencies.get(best_candidate_id, []):
                    if neighbor_id not in unit_assignments:
                        frontiers[category_name].add(neighbor_id)
        
        # If no progress, end growth phase
        if units_added_this_iteration == 0:
            print(f"  🛑 No progress in iteration {iteration}, ending controlled growth")
            break
    
    print(f"\n📊 CONTROLLED GROWTH COMPLETED:")
    print(f"   Iterations: {iteration}")
    print(f"   Units assigned: {len(unit_assignments)}/{total_units} ({len(unit_assignments)/total_units*100:.1f}%)")
    
    # OPTIMIZED FINAL DISTRIBUTION - much faster approach
    unassigned_units = [unit_id for unit_id in unit_lookup.keys() if unit_id not in unit_assignments]
    
    if unassigned_units and not check_timeout("final distribution"):
        print(f"\n📦 FAST DISTRIBUTION OF {len(unassigned_units)} REMAINING UNITS...")
        print("-" * 50)
        
        # Simple strategy: assign to categories with largest deficits in chunks
        category_deficits = []
        for category in categories:
            category_name = category['name']
            current = region_areas[category_name]
            target = region_targets[category_name]
            deficit = max(0, target - current)
            if deficit > 0:
                category_deficits.append((deficit, category_name))
        
        category_deficits.sort(reverse=True)  # Largest deficit first
        
        if category_deficits:
            # Distribute proportionally without expensive proximity calculations
            total_deficit = sum(deficit for deficit, _ in category_deficits)
            
            assigned_count = 0
            for i, (deficit, category_name) in enumerate(category_deficits, 1):
                if assigned_count >= len(unassigned_units):
                    break
                
                # Calculate proportion based on deficit
                proportion = deficit / total_deficit if total_deficit > 0 else 1.0 / len(category_deficits)
                units_for_category = int(len(unassigned_units) * proportion)
                
                # Assign units to this category
                end_idx = min(assigned_count + units_for_category, len(unassigned_units))
                
                for unit_id in unassigned_units[assigned_count:end_idx]:
                    unit_assignments[unit_id] = category_name
                    unit = unit_lookup[unit_id]
                    region_areas[category_name] += unit['area_km2']
                
                units_assigned = end_idx - assigned_count
                assigned_count = end_idx
                print(f"  {i:2}/{len(category_deficits)} 📌 {category_name:<30} - Assigned: {units_assigned:>4} units "
                      f"(deficit: {deficit:>6.0f} km²)")
            
            # Assign any remaining units to the largest deficit category
            if assigned_count < len(unassigned_units):
                if category_deficits:
                    largest_deficit_category = category_deficits[0][1]
                    remaining_units = unassigned_units[assigned_count:]
                    
                    for unit_id in remaining_units:
                        unit_assignments[unit_id] = largest_deficit_category
                        unit = unit_lookup[unit_id]
                        region_areas[largest_deficit_category] += unit['area_km2']
                    
                    print(f"  ⬆️  {largest_deficit_category:<30} - Assigned: {len(remaining_units):>4} remaining units")
    
    end_time = time.time()
    print(f"\n⏱️  CONTROLLED REGION GROWING COMPLETED:")
    print(f"   Duration: {end_time - start_time:.2f} seconds")
    print(f"   Iterations: {iteration}")
    print(f"   Final assignment: {len(unit_assignments)}/{total_units} units ({len(unit_assignments)/total_units*100:.1f}%)")
    
    # Report final areas
    print(f"\n📋 FINAL AREA SUMMARY AFTER CONTROLLED GROWTH:")
    print(f"{'Category':<35} {'Current':<12} {'Target':<12} {'Diff %':<10} {'Status':<10}")
    print("-" * 85)
    
    for category in categories:
        category_name = category['name']
        current = region_areas[category_name]
        target = region_targets[category_name]
        diff_pct = ((current - target) / target) * 100 if target > 0 else 0
        
        min_target = target * (1 - AREA_TOLERANCE_PCT)
        max_target = target * (1 + AREA_TOLERANCE_PCT)
        status = "✅ PASS" if min_target <= current <= max_target else "❌ FAIL"
        
        print(f"{category_name:<35} {current:<12.2f} {target:<12.2f} {diff_pct:<10.2f} {status:<10}")
    
    return unit_assignments, region_areas

def calculate_controlled_candidate_score(candidate_unit, category_name, current_area, target_area, adjacencies, unit_assignments):
    """Calculate score for controlled growth"""
    unit_area = candidate_unit['area_km2']
    new_area = current_area + unit_area
    
    # Area fit score - strongly prefer not overshooting
    area_ratio = new_area / target_area if target_area > 0 else 1
    if area_ratio <= 1.0:
        area_score = 2.0 - abs(area_ratio - 0.8)  # Prefer getting to 80% of target
    elif area_ratio <= 1.03:  # Within tolerance
        area_score = 1.0
    else:  # Would overshoot
        area_score = 0.01  # Very low score
    
    # Connectivity score - essential for contiguity
    neighbors_in_region = 0
    total_neighbors = len(adjacencies.get(candidate_unit['unit_id'], []))
    
    for neighbor_id in adjacencies.get(candidate_unit['unit_id'], []):
        if unit_assignments.get(neighbor_id) == category_name:
            neighbors_in_region += 1
    
    # Must have at least one neighbor in the region for contiguity
    if neighbors_in_region == 0:
        return 0  # Cannot maintain contiguity
    
    connectivity_score = (neighbors_in_region + 1) / (total_neighbors + 1)
    
    return area_score * connectivity_score * 10  # Boost good candidates

def create_contiguous_regions_from_assignments(units_gdf, unit_assignments, categories):
    """Create region geometries from unit assignments with detailed progress"""
    print("\n🗺️  CREATING CONTIGUOUS REGION GEOMETRIES FROM ASSIGNMENTS...")
    print("-" * 70)
    start_time = time.time()
    
    regions_data = []
    
    print(f"📊 Processing {len(categories)} categories...")
    
    for i, category in enumerate(categories, 1):
        category_name = category['name']
        print(f"\n🔧 {i:2}/{len(categories)} Creating region for {category_name}...")
        
        # Get all units assigned to this category
        assigned_unit_ids = [unit_id for unit_id, cat in unit_assignments.items() 
                            if cat == category_name]
        
        if not assigned_unit_ids:
            print(f"     ⚠️  No units assigned to {category_name}")
            continue
        
        print(f"     📊 Found {len(assigned_unit_ids)} assigned units")
        
        # Get geometries for assigned units
        assigned_units = units_gdf[units_gdf['unit_id'].isin(assigned_unit_ids)]
        
        if assigned_units.empty:
            print(f"     ❌ No matching geometries found for {category_name}")
            continue
        
        # Union all geometries
        print(f"     🔗 Merging {len(assigned_units)} unit geometries...")
        merged_geometry = assigned_units.geometry.unary_union
        
        # Calculate total area
        total_area = assigned_units['area_km2'].sum()
        
        # Analyze contiguity
        if hasattr(merged_geometry, 'geoms'):
            components = len(merged_geometry.geoms)
            geometry_type = "MultiPolygon"
        else:
            components = 1
            geometry_type = "Polygon"
        
        target_area = category['area_km2']
        area_diff_pct = ((total_area - target_area) / target_area) * 100 if target_area > 0 else 0
        
        print(f"     ✅ Created region: {geometry_type} with {components} components, {total_area:.2f} km²")
        print(f"     📊 Target: {target_area:.2f} km² (diff: {area_diff_pct:+.2f}%)")
        
        regions_data.append({
            'category': category_name,
            'geometry': merged_geometry,
            'area_km2': total_area,
            'components': components,
            'color': category.get('color', '#CCCCCC')
        })
    
    # Create GeoDataFrame
    print(f"\n📋 Creating final GeoDataFrame with {len(regions_data)} regions...")
    regions_gdf = gpd.GeoDataFrame(regions_data, crs=units_gdf.crs)
    
    end_time = time.time()
    print(f"\n✅ GEOMETRY CREATION COMPLETED:")
    print(f"   Duration: {end_time - start_time:.2f} seconds")
    print(f"   Regions created: {len(regions_gdf)}")
    print(f"   Total area: {regions_gdf['area_km2'].sum():.2f} km²")
    
    # Quick contiguity summary
    contiguous_count = sum(1 for _, region in regions_gdf.iterrows() 
                          if region['components'] == 1)
    print(f"   Contiguous regions: {contiguous_count}/{len(regions_gdf)} ({contiguous_count/len(regions_gdf)*100:.1f}%)")
    
    return regions_gdf

def export_and_plot(regions_gdf, base_filename):
    """Export results and create visualization with improved styling"""
    print(f"\n📤 EXPORTING AND PLOTTING RESULTS...")
    print("-" * 50)
    
    if regions_gdf.empty:
        print("No regions to export")
        return
    
    # Export to GeoJSON
    geojson_filename = f"{base_filename}.geojson"
    try:
        export_gdf = regions_gdf.copy()
        if 'area_km2_summed' in export_gdf.columns and 'area_km2_geom' in export_gdf.columns:
            del export_gdf['area_km2_summed']  # Keep geom area as primary
        export_gdf.to_file(geojson_filename, driver="GeoJSON")
        print(f"Regions exported to {geojson_filename}")
    except Exception as e:
        print(f"Error exporting to GeoJSON: {e}")
    
    # Create visualization
    print(f"Creating visualization: {base_filename}.png")
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot each region with its color
    for _, region in regions_gdf.iterrows():
        category_name = region['category']
        color = region['color']
        
        # Create temporary GeoDataFrame for this region
        temp_gdf = gpd.GeoDataFrame([region], crs=regions_gdf.crs)
        temp_gdf.plot(ax=ax, color=color, edgecolor='darkgrey', linewidth=0.5, alpha=0.8)
    
    ax.set_title("Belgium Land Use Regions (Contiguous Algorithm)", fontsize=18, fontweight='bold')
    ax.set_xlabel("Easting (m, EPSG:3035)", fontsize=14)
    ax.set_ylabel("Northing (m, EPSG:3035)", fontsize=14)
    
    # Create legend
    import matplotlib.patches as mpatches
    legend_patches = []
    for _, region in regions_gdf.iterrows():
        patch = mpatches.Patch(color=region['color'], label=region['category'])
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, title="Land Use Category", 
             loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=12, title_fontsize=14)
    
    plt.tight_layout()
    
    png_filename = f"{base_filename}.png"
    try:
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        print(f"Map saved to {png_filename}")
    except Exception as e:
        print(f"Error saving PNG: {e}")
    
    plt.close(fig)

def check_contiguity_and_results(regions_gdf, categories, original_total_area):
    """STRICT validation - ALL categories must be contiguous AND pass area tolerance"""
    print("\n" + "="*90)
    print("STRICT CONTIGUOUS RESULTS VALIDATION")
    print("="*90)
    
    # Track validation results
    contiguity_passed = 0
    area_passed = 0
    total_categories = len(categories)
    
    print("🔍 CONTIGUITY ANALYSIS (STRICT - ALL MUST BE SINGLE POLYGONS):")
    print("-" * 70)
    
    for _, region in regions_gdf.iterrows():
        category_name = region['category']
        geometry = region['geometry']
        area = region['area_km2']
        
        # Check if it's a single polygon (contiguous)
        if geometry.geom_type == 'Polygon':
            components = 1
            status = "✅ PASS"
            contiguity_passed += 1
        else:  # MultiPolygon
            if hasattr(geometry, 'geoms'):
                components = len(geometry.geoms)
            else:
                components = 1
            status = "❌ FAIL"
        
        print(f"  {category_name:<35} {components:>3} components - {status}")
    
    print(f"\n📊 CONTIGUITY SUMMARY: {contiguity_passed}/{total_categories} categories are contiguous")
    
    print(f"\n🎯 AREA ACCURACY ANALYSIS (STRICT - ALL MUST PASS ±{AREA_TOLERANCE_PCT*100:.1f}% TOLERANCE):")
    print("-" * 85)
    print(f"{'Category':<35} {'Target':<12} {'Actual':<12} {'Diff %':<10} {'Status':<10}")
    print("-" * 85)
    
    for category in categories:
        category_name = category['name']
        target_area = category['area_km2']
        
        # Find the region for this category
        region_row = regions_gdf[regions_gdf['category'] == category_name]
        if len(region_row) == 0:
            print(f"{category_name:<35} {'MISSING':<12} {'MISSING':<12} {'N/A':<10} {'❌ FAIL':<10}")
            continue
        
        actual_area = region_row.iloc[0]['area_km2']
        diff_pct = ((actual_area - target_area) / target_area) * 100 if target_area > 0 else 0
        
        min_target = target_area * (1 - AREA_TOLERANCE_PCT)
        max_target = target_area * (1 + AREA_TOLERANCE_PCT)
        
        if min_target <= actual_area <= max_target:
            status = "✅ PASS"
            area_passed += 1
        else:
            status = "❌ FAIL"
        
        print(f"{category_name:<35} {target_area:<12.2f} {actual_area:<12.2f} {diff_pct:<10.2f} {status:<10}")
    
    print("-" * 85)
    total_allocated = regions_gdf['area_km2'].sum()
    coverage_pct = (total_allocated / original_total_area) * 100 if original_total_area > 0 else 0
    overall_diff = ((total_allocated - sum(cat['area_km2'] for cat in categories)) / 
                   sum(cat['area_km2'] for cat in categories)) * 100
    
    print(f"{'TOTAL':<35} {sum(cat['area_km2'] for cat in categories):<12.2f} {total_allocated:<12.2f} {overall_diff:<10.2f} {'INFO':<10}")
    
    print(f"\n📊 AREA ACCURACY SUMMARY: {area_passed}/{total_categories} categories pass tolerance")
    print(f"📊 Original boundary area: {original_total_area:.2f} km²")
    print(f"📊 Coverage: {coverage_pct:.2f}%")
    
    # STRICT FINAL ASSESSMENT
    print("\n" + "="*90)
    if contiguity_passed == total_categories and area_passed == total_categories:
        print("🎉 STRICT VALIDATION PASSED - ALL CATEGORIES CONTIGUOUS AND ACCURATE! 🎉")
        print("="*90)
        return True
    else:
        print("❌ STRICT VALIDATION FAILED:")
        if contiguity_passed < total_categories:
            fragmented = total_categories - contiguity_passed
            print(f"   📍 CONTIGUITY: {fragmented} categories are fragmented (need single polygons)")
        if area_passed < total_categories:
            inaccurate = total_categories - area_passed
            print(f"   📏 AREA ACCURACY: {inaccurate} categories outside ±{AREA_TOLERANCE_PCT*100:.1f}% tolerance")
        print(f"   🎯 REQUIRED: ALL {total_categories} categories must be contiguous AND accurate")
        print("="*90)
        return False

def main():
    """Main execution function for contiguous region creation with timeout protection"""
    global START_TIME
    START_TIME = time.time()
    
    # Set up timeout protection
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    
    try:
        print("🇧🇪 FAST CONTIGUOUS BELGIUM LAND DIVIDER")
        print("="*50)
        print(f"Target categories: {len(LAND_USE_CATEGORIES)}")
        print(f"Total target area: {TOTAL_ALLOCATED_AREA_KM2:.2f} km²")
        print(f"Area tolerance: ±{AREA_TOLERANCE_PCT*100:.1f}%")
        print(f"⏰ TIMEOUT: {TIMEOUT_SECONDS} seconds (2 minutes)")
        print("Goal: Each category forms ONE contiguous region")
        
        # Step 1: Load or prepare units
        units_file = "geodata/prepared_units.geojson"
        
        if os.path.exists(units_file):
            print(f"\nLoading prepared units from {units_file}...")
            units_gdf = gpd.read_file(units_file)
            print(f"Loaded {len(units_gdf)} units")
            original_total_area = units_gdf['area_km2'].sum()
        else:
            print(f"\n❌ ERROR: Required file {units_file} not found!")
            print(f"📁 Please ensure the 'geodata/' folder contains the required files:")
            print(f"   - prepared_units.geojson (207 MB)")
            print(f"   - sh_statbel_statistical_sectors_31370_20240101.geojson.zip (43 MB)")
            print(f"\n💡 See README.md for download instructions and data sources.")
            return
        
        if check_timeout("initialization"):
            print("⏰ Timeout during initialization")
            return
        
        # Step 2: Build efficient adjacency
        print(f"\n⏰ Time remaining: {TIMEOUT_SECONDS - (time.time() - START_TIME):.1f}s")
        adjacencies = build_efficient_adjacency(units_gdf)
        
        if check_timeout("adjacency calculation"):
            print("⏰ Timeout after adjacency calculation")
            return
        
        # Step 3: Select optimal seeds
        print(f"\n⏰ Time remaining: {TIMEOUT_SECONDS - (time.time() - START_TIME):.1f}s")
        seeds = select_optimal_seeds(units_gdf, LAND_USE_CATEGORIES, adjacencies)
        
        if check_timeout("seed selection"):
            print("⏰ Timeout after seed selection")
            return
        
        # Step 4: Controlled contiguous region growing
        print(f"\n⏰ Time remaining: {TIMEOUT_SECONDS - (time.time() - START_TIME):.1f}s")
        unit_assignments, region_areas = controlled_contiguous_region_growing(
            units_gdf, LAND_USE_CATEGORIES, adjacencies, seeds)
        
        if check_timeout("region growing"):
            print("⏰ Timeout after region growing")
            # Continue with current results
        
        # Step 5: Create contiguous region geometries
        print(f"\n⏰ Time remaining: {TIMEOUT_SECONDS - (time.time() - START_TIME):.1f}s")
        regions_gdf = create_contiguous_regions_from_assignments(
            units_gdf, unit_assignments, LAND_USE_CATEGORIES)
        
        if check_timeout("geometry creation"):
            print("⏰ Timeout after geometry creation")
            # Continue with current results
        
        # Step 6: Export and visualize (skip if approaching timeout)
        if not check_timeout("export"):
            print(f"\n⏰ Time remaining: {TIMEOUT_SECONDS - (time.time() - START_TIME):.1f}s")
            export_and_plot(regions_gdf, "belgium_contiguous_regions_fast")
        else:
            print("⏰ Skipping export due to timeout - saving basic results only")
            try:
                regions_gdf.to_file("belgium_contiguous_regions_fast.geojson", driver="GeoJSON")
                print("Saved basic results to belgium_contiguous_regions_fast.geojson")
            except:
                pass
        
        # Step 7: Validate contiguity and results
        if not check_timeout("validation"):
            check_contiguity_and_results(regions_gdf, LAND_USE_CATEGORIES, original_total_area)
        else:
            print("⏰ Skipping detailed validation due to timeout")
            
        total_time = time.time() - START_TIME
        print(f"\n🎉 Fast processing complete in {total_time:.1f} seconds! ⚡")
        
        if total_time < TIMEOUT_SECONDS:
            print(f"✅ Completed within timeout limit ({TIMEOUT_SECONDS}s)")
        else:
            print(f"⚠️  Exceeded timeout but results available")
    
    except TimeoutException:
        print(f"\n⏰ TIMEOUT: Algorithm terminated after {TIMEOUT_SECONDS} seconds")
        print("Partial results may be available in working files")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cancel the alarm
        signal.alarm(0)

if __name__ == "__main__":
    main() 