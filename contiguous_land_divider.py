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
    """Select optimal seed units for each category based on connectivity and spatial distribution"""
    print("🌱 SELECTING OPTIMAL SEEDS FOR REGION GROWING...")
    print("-" * 60)
    start_time = time.time()
    
    # Create efficient lookup for unit centroids and bounds
    print(f"📊 Preparing seed selection for {len(categories)} categories...")
    unit_centroids = {row['unit_id']: row.geometry.centroid for _, row in units_gdf.iterrows()}
    total_area = units_gdf['area_km2'].sum()
    
    # Get Belgium bounds for spatial distribution
    min_x, min_y, max_x, max_y = units_gdf.total_bounds
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    
    seeds = {}
    used_seeds = set()  # Track used seeds to avoid duplicates
    
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
        
        # Use different random seed for each category to avoid same samples
        sample_size = min(2000, len(units_gdf))  # Increased sample size
        np.random.seed(42 + i)  # Different seed for each category
        sample_indices = np.random.choice(len(units_gdf), size=sample_size, replace=False)
        sample_units = units_gdf.iloc[sample_indices]
        
        # Define target region center based on area fraction and category index
        # Distribute categories spatially to avoid clustering
        angle = (i - 1) * (2 * np.pi / len(categories))  # Distribute around circle
        target_radius = 50000 * np.sqrt(area_fraction)  # Larger categories toward center
        target_x = center_x + target_radius * np.cos(angle)
        target_y = center_y + target_radius * np.sin(angle)
        
        for _, unit in sample_units.iterrows():
            unit_id = unit['unit_id']
            
            # Skip already used seeds
            if unit_id in used_seeds:
                continue
                
            unit_area = unit['area_km2']
            
            # Calculate connectivity score
            neighbor_count = len(adjacencies.get(unit_id, []))
            connectivity_score = min(neighbor_count / 8.0, 1.0)  # Normalize to 0-1
            
            # Calculate spatial distribution score - prefer units near target location
            unit_centroid = unit_centroids[unit_id]
            if hasattr(unit_centroid, 'x') and hasattr(unit_centroid, 'y'):
                distance_to_target = np.sqrt((unit_centroid.x - target_x)**2 + (unit_centroid.y - target_y)**2)
                # Convert distance score (closer = better)
                max_distance = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
                spatial_score = max(0, 1.0 - (distance_to_target / max_distance))
            else:
                spatial_score = 0.5  # Default if geometry issues
            
            # Area appropriateness score
            area_score = min(unit_area / 5.0, 1.0)  # Prefer reasonable sized units
            
            # Category-specific scoring
            if target_area > 5000:  # Very large categories (Agricultural, Forest)
                final_score = connectivity_score * 3.0 + spatial_score * 2.0 + area_score
            elif target_area > 1000:  # Large categories 
                final_score = connectivity_score * 2.0 + spatial_score * 3.0 + area_score
            else:  # Small categories
                final_score = connectivity_score * 1.5 + spatial_score * 1.0 + area_score * 2.0
            
            # Bonus for well-connected units
            if neighbor_count >= 6:
                final_score *= 1.2
            
            # Penalty for very small units for large categories
            if target_area > 1000 and unit_area < 0.5:
                final_score *= 0.5
            
            if final_score > best_score:
                best_score = final_score
                best_unit_id = unit_id
                best_unit_area = unit_area
            
            candidates_evaluated += 1
        
        if best_unit_id is not None:
            seeds[category_name] = best_unit_id
            used_seeds.add(best_unit_id)  # Mark as used
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
    print(f"   Unique seeds: {len(set(seeds.values()))} (should equal {len(seeds)})")
    
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
            
            # Tighter area control for better accuracy - different limits by category size
            if target_area > 1000:  # Large categories - allow 2% overshoot
                max_overshoot = 1.02
            else:  # Small categories - allow 3% overshoot (they need more flexibility)
                max_overshoot = 1.03
                
            # Only consider categories that haven't exceeded their limit
            if completion_ratio < max_overshoot:
                # Enhanced priority calculation
                if completion_ratio < 0.97:  # Still significantly under target
                    priority = (target_area - current_area) / target_area * 2.0  # High priority
                elif completion_ratio < 1.0:  # Close to target but under
                    priority = (target_area - current_area) / target_area * 1.5  # Medium priority  
                else:  # Over target but within limit
                    priority = (target_area - current_area) / target_area * 0.5  # Low priority
                    
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
    
    # IMPROVED FINAL DISTRIBUTION - smarter approach for area accuracy
    unassigned_units = [unit_id for unit_id in unit_lookup.keys() if unit_id not in unit_assignments]
    
    if unassigned_units and not check_timeout("final distribution"):
        print(f"\n📦 SMART DISTRIBUTION OF {len(unassigned_units)} REMAINING UNITS...")
        print("-" * 50)
        
        # Calculate area deficits and priorities
        category_needs = []
        for category in categories:
            category_name = category['name']
            current = region_areas[category_name]
            target = region_targets[category_name]
            
            # Calculate both absolute and percentage deficit
            absolute_deficit = max(0, target - current)
            percentage_deficit = (target - current) / target if target > 0 else 0
            
            # Priority scoring: weight by both deficit size and percentage
            if absolute_deficit > 0:
                if target > 1000:  # Large categories - prioritize absolute deficit
                    priority_score = absolute_deficit * 0.7 + percentage_deficit * target * 0.3
                else:  # Small categories - prioritize percentage deficit
                    priority_score = absolute_deficit * 0.3 + percentage_deficit * target * 0.7
                
                category_needs.append((priority_score, absolute_deficit, category_name, current, target))
        
        category_needs.sort(reverse=True)  # Highest priority first
        
        if category_needs:
            print(f"     📊 Distribution priorities:")
            for i, (priority, deficit, cat_name, current, target) in enumerate(category_needs[:5], 1):
                pct_deficit = ((target - current) / target * 100) if target > 0 else 0
                print(f"     {i}. {cat_name:<30} deficit: {deficit:>6.1f} km² ({pct_deficit:>5.1f}%)")
            
            # Smart assignment strategy: assign to categories with borders near unassigned units
            assigned_count = 0
            assignment_attempts = 0
            max_attempts = len(unassigned_units) * 3  # Prevent infinite loops
            
            # Create lookup of which categories border which unassigned units
            unassigned_borders = {}
            for unit_id in unassigned_units:
                bordering_categories = set()
                for neighbor_id in adjacencies.get(unit_id, []):
                    neighbor_cat = unit_assignments.get(neighbor_id)
                    if neighbor_cat:
                        bordering_categories.add(neighbor_cat)
                unassigned_borders[unit_id] = bordering_categories
            
            # Process units in smart order: those with fewer bordering categories first (easier to assign)
            sorted_unassigned = sorted(unassigned_units, 
                                     key=lambda uid: len(unassigned_borders.get(uid, set())))
            
            for unit_id in sorted_unassigned:
                if assigned_count >= len(unassigned_units) or assignment_attempts >= max_attempts:
                    break
                
                assignment_attempts += 1
                unit = unit_lookup[unit_id]
                unit_area = unit['area_km2']
                
                # Find best category for this unit
                best_category = None
                best_score = -1
                
                bordering_cats = unassigned_borders.get(unit_id, set())
                
                # Consider categories that: 1) border this unit, 2) have deficit, 3) are high priority
                candidate_categories = []
                for priority, deficit, cat_name, current, target in category_needs:
                    if deficit > 0:  # Still has deficit
                        new_area = region_areas[cat_name] + unit_area
                        new_deficit = max(0, target - new_area)
                        
                        # Calculate assignment score
                        if cat_name in bordering_cats:
                            # Bordering category - good for contiguity
                            border_bonus = 3.0
                        else:
                            # Non-bordering - small penalty but still possible
                            border_bonus = 0.5
                        
                        # Area improvement score
                        area_improvement = deficit - new_deficit
                        area_score = area_improvement / target if target > 0 else 0
                        
                        # Deficit urgency score
                        deficit_score = deficit / target if target > 0 else 0
                        
                        final_score = border_bonus * area_score * (1.0 + deficit_score)
                        
                        candidate_categories.append((final_score, cat_name, new_deficit))
                
                # Select best category
                if candidate_categories:
                    candidate_categories.sort(reverse=True)  # Highest score first
                    best_score, best_category, remaining_deficit = candidate_categories[0]
                
                # Assign to best category
                if best_category and best_score > 0:
                    unit_assignments[unit_id] = best_category
                    region_areas[best_category] += unit_area
                    assigned_count += 1
                    
                    # Update category_needs list
                    category_needs = [(p, max(0, region_targets[cn] - region_areas[cn]), cn, 
                                     region_areas[cn], region_targets[cn]) 
                                    for p, d, cn, c, t in category_needs]
                    category_needs = [item for item in category_needs if item[1] > 0]
                    category_needs.sort(reverse=True)
            
            print(f"     ✅ Smart assignment completed: {assigned_count}/{len(unassigned_units)} units")
            
            # Assign any remaining units to largest deficit categories
            remaining_unassigned = [uid for uid in unassigned_units if uid not in unit_assignments]
            if remaining_unassigned and category_needs:
                fallback_category = category_needs[0][2]  # Largest deficit
                fallback_count = 0
                
                for unit_id in remaining_unassigned:
                    unit_assignments[unit_id] = fallback_category
                    unit = unit_lookup[unit_id]
                    region_areas[fallback_category] += unit['area_km2']
                    fallback_count += 1
                
                print(f"     📌 Fallback assignment: {fallback_count} units to {fallback_category}")
        else:
            print(f"     ⚠️  No categories with deficits found")
    
    end_time = time.time()
    print(f"\n⏱️  CONTROLLED REGION GROWING COMPLETED:")
    print(f"   Duration: {end_time - start_time:.2f} seconds")
    print(f"   Iterations: {iteration}")
    print(f"   Final assignment: {len(unit_assignments)}/{total_units} units ({len(unit_assignments)/total_units*100:.1f}%)")
    
    # Report final areas
    print(f"\n📋 FINAL AREA SUMMARY AFTER CONTROLLED GROWTH:")
    print(f"{'Category':<35} {'Current':<12} {'Target':<12} {'Diff %':<10} {'Status':<10}")
    print("-" * 85)
    
    categories_within_tolerance = 0
    for category in categories:
        category_name = category['name']
        current = region_areas[category_name]
        target = region_targets[category_name]
        diff_pct = ((current - target) / target) * 100 if target > 0 else 0
        
        min_target = target * (1 - AREA_TOLERANCE_PCT)
        max_target = target * (1 + AREA_TOLERANCE_PCT)
        status = "✅ PASS" if min_target <= current <= max_target else "❌ FAIL"
        if status == "✅ PASS":
            categories_within_tolerance += 1
        
        print(f"{category_name:<35} {current:<12.2f} {target:<12.2f} {diff_pct:<10.2f} {status:<10}")
    
    print(f"\n📊 AREA SUMMARY: {categories_within_tolerance}/{len(categories)} categories within ±{AREA_TOLERANCE_PCT*100:.1f}% tolerance")
    
    return unit_assignments, region_areas

def calculate_controlled_candidate_score(candidate_unit, category_name, current_area, target_area, adjacencies, unit_assignments):
    """Calculate score for controlled growth with enhanced contiguity and area accuracy"""
    unit_area = candidate_unit['area_km2']
    new_area = current_area + unit_area
    
    # Area fit score - strongly prioritize getting close to target without overshooting
    area_ratio = new_area / target_area if target_area > 0 else 1
    if area_ratio <= 0.95:  # Still significantly under target
        # Encourage steady growth toward target
        progress_to_target = current_area / target_area if target_area > 0 else 0
        area_score = 4.0 + (1.0 - abs(area_ratio - 0.85))  # Prefer getting to 85% of target
    elif area_ratio <= 0.97:  # Getting close but still under
        area_score = 5.0 + (1.0 - abs(area_ratio - 0.96))  # High score for getting close
    elif area_ratio <= 1.0:  # Very close to perfect target
        area_score = 8.0 - abs(area_ratio - 1.0) * 20  # Peak score at exactly target
    elif area_ratio <= 1.01:  # Slightly over but acceptable
        area_score = 6.0 - abs(area_ratio - 1.0) * 15  # Good score but penalized
    elif area_ratio <= 1.03:  # Within tolerance zone but over
        area_score = 3.0 - abs(area_ratio - 1.0) * 10  # Lower score to discourage
    else:  # Would overshoot tolerance significantly
        area_score = 0.001  # Very low score to avoid overshooting
    
    # Connectivity score - CRITICAL for contiguity
    neighbors_in_region = 0
    neighbors_in_other_regions = 0
    total_neighbors = len(adjacencies.get(candidate_unit['unit_id'], []))
    
    for neighbor_id in adjacencies.get(candidate_unit['unit_id'], []):
        neighbor_assignment = unit_assignments.get(neighbor_id)
        if neighbor_assignment == category_name:
            neighbors_in_region += 1
        elif neighbor_assignment is not None:
            neighbors_in_other_regions += 1
    
    # Must have at least one neighbor in the region for contiguity
    if neighbors_in_region == 0:
        return 0.001  # Almost zero - cannot maintain contiguity
    
    # Prefer units with multiple connections to the region (stronger contiguity)
    connectivity_score = neighbors_in_region
    if neighbors_in_region >= 2:
        connectivity_score *= 1.5  # Bonus for strong connectivity
    if neighbors_in_region >= 3:
        connectivity_score *= 1.3  # Extra bonus for very strong connectivity
    
    # Penalty for creating complex boundaries (many neighbors in other regions)
    if neighbors_in_other_regions > 0:
        boundary_penalty = 1.0 / (1.0 + neighbors_in_other_regions * 0.3)
    else:
        boundary_penalty = 1.2  # Bonus for internal growth
    
    # Compactness bonus - prefer units that don't create long tendrils
    compactness_score = 1.0
    if total_neighbors > 0:
        # Higher ratio of same-region neighbors = more compact
        same_region_ratio = neighbors_in_region / total_neighbors
        compactness_score = 1.0 + same_region_ratio * 0.5
    
    # Size appropriateness - prefer reasonable unit sizes
    size_score = 1.0
    if target_area > 1000:  # Large categories
        if unit_area < 0.1:  # Very small units
            size_score = 0.7
        elif unit_area > 20:  # Very large units
            size_score = 0.8
    else:  # Small categories
        if unit_area > 10:  # Large units for small categories
            size_score = 0.6
    
    # Category-specific bonuses
    category_bonus = 1.0
    if target_area > 5000:  # Very large categories (Agricultural, Forest)
        # Prioritize area accuracy more for large categories
        category_bonus = 1.0 + (area_score / 10.0)
    elif target_area < 50:  # Very small categories
        # Prioritize any growth for tiny categories
        if area_ratio < 0.5:
            category_bonus = 2.0  # Strong bonus to help small categories grow
    
    # Final score calculation
    final_score = (area_score * connectivity_score * boundary_penalty * 
                  compactness_score * size_score * category_bonus)
    
    # Debug logging for problematic cases (occasionally)
    if np.random.random() < 0.001:  # 0.1% chance to log
        print(f"    🔍 Unit {candidate_unit['unit_id']}: area={area_score:.2f}, "
              f"connect={connectivity_score:.2f}, boundary={boundary_penalty:.2f}, "
              f"final={final_score:.2f}")
    
    return final_score

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
            print(f"   📍 AREA ACCURACY: {inaccurate} categories outside ±{AREA_TOLERANCE_PCT*100:.1f}% tolerance")
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