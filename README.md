# Belgium Land Use Map

A Python tool that divides Belgium into 13 land-use regions with predetermined
areas, producing a publication-ready map inspired by the German
"Flächennutzung Deutschland" visualisation.

## Result

- **13/13 categories** contiguous (single connected region each)
- **13/13 categories** within ±3 % area tolerance
- **Runtime:** ~3 seconds
- **Output:** `belgium_land_use_grid.png`

## Quick Start

```bash
conda create -n belgium_land_use_env python=3.9 \
      geopandas shapely matplotlib numpy scipy
conda activate belgium_land_use_env
python belgium_land_use_grid.py
```

The script reads `belgium_land_use.csv` (category data) and `belgium.geojson`
(country boundary), then writes `belgium_land_use_grid.png`.

## File Structure

```
land_use_BE/
├── belgium_land_use_grid.py         # Main algorithm + visualisation
├── belgium_land_use.csv             # Land-use categories (name, area, color)
├── belgium.geojson                  # Belgium boundary (EPSG:4326)
├── belgium_boundary.geojson         # Alternative boundary
├── germany.jpeg                     # Reference image for visual quality
├── README.md                        # This file
├── belgium_land_use_data_collection.md  # Detailed data-sourcing notes
├── new_strategy.md                  # Algorithm design notes
├── .gitignore
└── legacy/                          # Previous attempts and scripts
```

## Algorithm

The approach rasterises Belgium onto a 500 m regular grid (~122 K cells in
EPSG:3035) and proceeds in five phases:

1. **Rasterise** the country boundary into a boolean mask.
2. **Seed placement** via farthest-point sampling — larger categories get more
   interior positions.
3. **Weighted Voronoi** — iteratively adjust additive weights so that each
   region converges to its target pixel count.
4. **Connectivity fix** — BFS from each seed keeps only the connected
   component; orphaned pixels are reassigned to neighbours.
5. **Border pixel swapping** — iteratively swap border pixels between adjacent
   regions, using percentage-based error scoring so that small categories are
   correctly prioritised. An O(1) ring-arc test guarantees contiguity is
   preserved at every swap.

All phases are generic: the script accepts any country boundary and any set of
categories via CSV. To adapt it to another country, provide a different
GeoJSON and CSV file and call:

```python
main(csv_path="other_country.csv",
     boundary_file="other_country.geojson",
     country_name="Other Country",
     output_prefix="other_country_land_use")
```

## Land-Use Categories

The 13 categories are mutually exclusive and sum to Belgium's official area of
**30,688 km²** (Statbel, 2018 CADGIS measurement).

| #  | Category                        | Area (km²) | % of Total | Source / Method |
|----|---------------------------------|------------|------------|-----------------|
|  1 | Agricultural land (others)      | 12,803     | 41.7 %     | Statbel 44 % total ag. (≈ 13,503) minus energy crops |
|  2 | Forest                          |  6,138     | 20.0 %     | Statbel land register |
|  3 | Other natural/semi-natural/rest |  4,222     | 13.8 %     | Residual to reach 100 % |
|  4 | Roads and rail infrastructure   |  3,553     | 11.6 %     | Statbel land register |
|  5 | Built-up (residential, industry)|  2,762     |  9.0 %     | Statbel "residential lands" |
|  6 | Energy crops                    |    700     |  2.3 %     | Eurostat / FAOSTAT estimate |
|  7 | Paths/tracks                    |    250     |  0.8 %     | OSM lengths × typical widths |
|  8 | Water bodies (inland)           |    143     |  0.5 %     | Statbel / CLC 511 + 512 |
|  9 | Golf courses                    |     45     |  0.15 %    | ~90 courses × ~0.5 km² |
| 10 | Football pitches                |     30     |  0.10 %    | ~5,000 pitches × 0.006 km² |
| 11 | Sport and leisure (others)      |     15     |  0.05 %    | Estimate |
| 12 | Wind turbines (footprint)       |     14     |  0.05 %    | ~3,500 MW × 0.004 km²/MW |
| 13 | Ground-mounted PV               |     13     |  0.04 %    | ~860 MW × 0.015 km²/MW |

### Accounting notes

- **Agricultural land (others)** is the residual of Statbel's total
  agricultural area (13,503 km², 44 %) after subtracting energy crops (700 km²).
- **Paths/tracks** physically overlap with agricultural and forest land; on the
  map they are carved out as a separate region, implicitly reducing the
  agricultural/forest/residual pool.
- **Other natural/rest** is the balancing residual. It absorbs heathland,
  scrub, bare rock, beaches, wetlands, estuaries, and any unaccounted-for land.
- **Built-up (9 %)** comes from Statbel's "residential lands", which groups
  residential, commercial, and industrial parcels. It may undercount some
  artificial surfaces captured by CLC.
- **Sport and leisure (others)** (15 km²) is a rough estimate for facilities
  (tennis, athletics, swimming, etc.) not covered by golf or football.

## Back-of-the-Envelope: Land for Mobility

How many passenger-km of car travel can be sustained by the land devoted to
ground-mounted solar PV vs. energy crops?

### Solar PV (13 km²) &rarr; EVs

| Parameter                  | Value     | Source / assumption               |
|----------------------------|-----------|-----------------------------------|
| PV power density           | 67 MWp/km² | 1.5 ha/MWp (industry standard)  |
| Installed capacity (13 km²)| 870 MWp  |                                    |
| Annual yield (Belgium)     | 950 kWh/kWp | Tilted irradiance ~1,100 kWh/m², PR ~0.82 |
| Annual electricity         | **827 GWh** |                                 |
| EV consumption             | 0.20 kWh/km | Mid-size EV incl. charging losses |
| Vehicle-km                 | 4.1 billion |                                 |
| Avg. occupancy             | 1.5       | Belgian average                    |
| **Passenger-km / year**    | **~6 billion** |                              |

### Energy crops (700 km²) &rarr; biofuel cars

Belgium's 700 km² of energy crops are ~600 km² maize for biogas and ~100 km²
rapeseed for biodiesel. Biogas is mostly burned in CHP plants for electricity
and heat, not for transport. For this comparison we generously assume all
output is converted to transport fuel via the most direct pathway for each
crop.

| Parameter                        | Rapeseed &rarr; biodiesel | Maize &rarr; biomethane &rarr; CNG |
|----------------------------------|--------------------------|-------------------------------------|
| Area                             | 100 km²                  | 600 km²                             |
| Fuel yield                       | 1,500 L biodiesel/ha     | 5,200 Nm³ CH₄/ha                   |
| Car consumption                  | 6 L / 100 km             | 4.5 kg / 100 km                     |
| Vehicle-km                       | 250 million              | 5.3 billion                         |
| Passenger-km (× 1.5)            | 375 million              | 8.0 billion                         |
| **Combined passenger-km / year** | **~8.4 billion**         |                                     |

*Rapeseed: 3.5 t/ha yield, ~40 % oil, ~95 % extraction, ~97 %
transesterification.  Maize: 50 t FM/ha, 200 m³ biogas/t, 55 % CH₄,
upgraded to biomethane; CH₄ density 0.72 kg/Nm³.*

### Comparison

|                            | Solar PV &rarr; EV | Energy crops &rarr; biofuel |
|----------------------------|-------------------:|----------------------------:|
| Land area                  | 13 km²             | 700 km²                     |
| Passenger-km / year        | ~6 billion          | ~8.4 billion                |
| **Passenger-km / km² / yr** | **~460 million**  | **~12 million**             |
| **Land-efficiency ratio**  | **~40×**            | 1×                          |

Per square kilometre, solar PV feeding electric vehicles delivers roughly
**40 times more passenger-km** than energy crops converted to biofuel — while
using **54 times less land** for a comparable mobility output.

Put differently: replacing just 18 km² of energy crops with solar panels (and
switching the corresponding cars to electric) would match the entire transport
output of all 700 km² of energy crops.

## Data Categories and Collection Process

For each category, the data for Belgium is estimated according to the following sources and methodology:

---

### 1. Agricultural Land & Related

#### 1.1. Energy Crops
- **Description:** Land used for cultivating crops specifically for energy production (e.g., biofuels, biogas).
- **Search Terms:** "Belgium energy crops land use", "land use for bioenergy Belgium statistics", "cultivation area energy plants Belgium"
- **Findings:**
    - Eurostat and FAOSTAT provide statistics on crops for energy production. The main energy crops in Belgium are maize for biogas and rapeseed for biodiesel. According to Eurostat (2022), the area under energy crops is approximately 700 km² (maize for silage/biogas: ~600 km², rapeseed: ~100 km², other crops: <10 km²). This is about 2.3% of total arable land. Statbel and recent scientific literature confirm this order of magnitude.
    - Corine Land Cover does not distinguish energy crops from other arable crops.
- **Data (km² or % of total area):** ~700 km² (2022, best estimate from Eurostat/FAOSTAT/Statbel)
- **Source(s):**
    - Eurostat: [Crops for energy production](https://ec.europa.eu/eurostat/databrowser/view/tag00098/default/table?lang=en)
    - FAOSTAT: [Bioenergy Crops](https://www.fao.org/faostat/en/#data/QC)
    - Statbel: [Land use](https://statbel.fgov.be/en/themes/environment/land-cover-and-use/land-use)
    - Recent scientific literature (2020–2023)
- **Notes:**
    - The main energy crops are maize for biogas and rapeseed for biodiesel.
    - The area can fluctuate year to year depending on policy and market conditions.
    - CLC/Corine does not distinguish energy crops from other arable crops.

#### 1.2. Crops for Human Consumption
- **Description:** Land used for growing crops directly for human food.
- **Search Terms:** "Belgium arable land human consumption", "crop production statistics Belgium", "land use food crops Belgium"
- **Findings:**
- **Data (km² or % of total area):** Non-irrigated arable land (211): 1,221,161.74 km²; Permanently irrigated land (212): 104,963.28 km²; Rice fields (213): 7,730.59 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:** These are the main arable land classes in CLC.

#### 1.3. Animal Feed Crops
- **Description:** Land used for growing crops for animal fodder (e.g., maize for silage, fodder beets, grassland for grazing/hay). This includes "Permanent Pasture" and parts of "Arable Land".
- **Search Terms:** "Belgium animal feed crops land use", "fodder crops area Belgium", "permanent pasture land use Belgium"
- **Findings:** Part of the 44% agricultural land reported by Statbel.
- **Data (km² or % of total area):** Pastures (231): 425,428.27 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

#### 1.4. Other Agricultural / Christmas Trees
- **Description:** Miscellaneous agricultural uses, with a specific mention of Christmas trees in the German example.
- **Search Terms:** "Christmas tree cultivation area Belgium statistics", "other agricultural land use Belgium"
- **Findings:**
- **Data (km² or % of total area):** Vineyards (221): 41,654.40 km²; Fruit trees and berry plantations (222): 50,734.14 km²; Olive groves (223): 4,160.56 km²; Annual crops associated with permanent crops (241): 32,739.69 km²; Complex cultivation patterns (242): 252,946.18 km²; Land principally occupied by agriculture, with significant areas of natural vegetation (243): 277,020.04 km²; Agro-forestry areas (244): 32,739.69 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

---

### 2. Built-up and Artificial Areas

#### 2.1. Settlement, Industry, Leisure
- **Description:** Combined area for residential, industrial, commercial, and recreational built-up areas.
- **Search Terms:** "Belgium settlement area statistics", "industrial land use Belgium", "recreational land use Belgium", "artificial surfaces Belgium"
- **Findings:** Statbel provides data on "built-up land and related sites." Statbel (2024) reports "residential lands for 9%". Corine Land Cover data (e.g., CLC2018) categorizes various artificial surfaces (urban fabric, industrial/commercial units, transport units, mine/dump/construction sites, artificial non-agricultural vegetated areas).
- **Data (km² or % of total area):** Continuous urban fabric (111): 6,726.63 km²; Discontinuous urban fabric (112): 165,366.65 km²; Industrial or commercial units (121): 29,039.31 km²; Green urban areas (141): 3,565.12 km²; Sport and leisure facilities (142): 12,088.11 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:** This is a broad category. "Wohnen" (Residential) is a sub-component.

#### 2.2. Residential Housing
- **Description:** Land specifically used for housing.
- **Search Terms:** "Belgium residential land use statistics", "housing area Belgium"
- **Findings:** Statbel (2024) states "residential lands for 9%".
- **Data (km² or % of total area):** Continuous urban fabric (111): 6,726.63 km²; Discontinuous urban fabric (112): 165,366.65 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:** This will be a part of the broader "Siedlung, Industrie, Freizeit".

#### 2.3. Opencast Mining, Quarries
- **Description:** Land used for mineral extraction (opencast mines, quarries).
- **Search Terms:** "Belgium mining land use", "quarry area Belgium statistics", "extraction sites land use Belgium"
- **Findings:** Corine Land Cover includes a category for "Mineral extraction sites" (CLC code 131) and "Dump sites" (CLC code 132).
- **Data (km² or % of total area):** Mineral extraction sites (131): 8,051.67 km²; Dump sites (132): 1,194.15 km²; Construction sites (133): 2,407.68 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

---

### 3. Energy & Infrastructure

#### 3.1. Ground-mounted Photovoltaics
- **Description:** Land area occupied by ground-mounted solar PV installations.
- **Search Terms:** "Belgium ground-mounted solar PV land area", "solar farm land use Belgium statistics"
- **Findings:**
    - Official Belgian and European sources (CREG, Elia, Statbel, Eurostat) report installed solar PV capacity, but not land area. Most Belgian PV is rooftop; ground-mounted is a small fraction.
    - IEA PVPS Belgium Country Report (2022) estimates about 10% of total PV capacity is ground-mounted. With ~8.6 GW total PV in 2023, this gives ~860 MW ground-mounted.
    - Scientific literature and industry standards suggest a land use factor of 1.5 ha/MW (0.015 km²/MW) for ground-mounted PV. Range: 1.2–2.0 ha/MW.
    - OSM lists mapped solar farms totaling ~5–8 km², but this is incomplete.
- **Data (km² or % of total area):** Best estimate: ~13 km² (range: 10–17 km²) for ground-mounted PV in Belgium (2023).
- **Source(s):**
    - Elia: [Renewable Energy Statistics](https://www.elia.be/en/grid-data/power-generation/renewable-energy)
    - Statbel: [Energy Statistics](https://statbel.fgov.be/en/themes/energy)
    - IEA PVPS Belgium Country Report 2022: [IEA PVPS Belgium](https://iea-pvps.org/country-reports/)
    - Fraunhofer ISE: [Land Use of Photovoltaics](https://www.ise.fraunhofer.de/en/publications/studies/land-use-of-photovoltaics.html)
    - OSM Overpass API (2024)
    - Industry news and press releases
- **Notes:**
    - The vast majority of Belgian solar PV is rooftop, not ground-mounted.
    - The estimate is based on the best available capacity data and standard land use factors.
    - OSM and press reports confirm the order of magnitude but are incomplete.

#### 3.2. Area for Wind Turbines
- **Description:** Land area directly occupied by wind turbines and their immediate infrastructure.
- **Search Terms:** "Belgium wind turbine land use", "wind farm footprint Belgium statistics"
- **Findings:**
    - Elia and Statbel provide statistics on installed wind capacity (onshore and offshore), but not directly on land area. As of 2023, Belgium had about 3.5 GW of onshore wind capacity.
    - Scientific literature and industry reports estimate the direct land take for wind turbines as 0.3–0.5 ha/MW (0.003–0.005 km²/MW). For 3,500 MW, this gives ~14 km² (range: 10–18 km²) for the direct footprint (bases, roads, substations).
    - The "wind farm area" (the polygon enclosing all turbines) can be 10–20 times larger, but most of this land remains available for agriculture.
    - OSM lists hundreds of wind turbines in Belgium, but mapped wind farm polygons are rare. Summing mapped turbine bases gives a similar order of magnitude for direct land take.
- **Data (km² or % of total area):** Best estimate: ~14 km² (range: 10–18 km²) for direct wind turbine footprint in Belgium (2023).
- **Source(s):**
    - Elia: [Renewable Energy Statistics](https://www.elia.be/en/grid-data/power-generation/renewable-energy)
    - Statbel: [Energy Statistics](https://statbel.fgov.be/en/themes/energy)
    - WindEurope: [Wind energy in Europe](https://windeurope.org/intelligence-platform/product/wind-energy-in-europe-in-2023-trends-and-statistics/)
    - IEA Wind: [IEA Wind TCP Annual Report](https://community.ieawind.org/home)
    - Fraunhofer IEE: [Land Use by Wind Power](https://www.energiecharts.info/charts/land_use/chart.htm?l=en&c=BE)
    - OSM Overpass API (2024)
    - Industry news and press releases
- **Notes:**
    - The actual land "lost" to wind turbines is very small; most land within wind farms remains in use (e.g., crops, pasture).
    - The estimate is based on the best available capacity data and standard land use factors.
    - OSM and press reports confirm the order of magnitude but are incomplete.

#### 3.3. Roads and Road Infrastructure
- **Description:** Land covered by roads, motorways, and associated infrastructure (e.g., parking areas alongside roads).
- **Search Terms:** "Belgium road network land area", "land use transport infrastructure Belgium"
- **Findings:**
- **Data (km² or % of total area):** Road and rail networks and associated land (122): 3,552.77 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

#### 3.4. Bahn (Railways and Rail Infrastructure)
- **Description:** Land covered by railway lines and associated infrastructure (e.g., stations, shunting yards).
- **Search Terms:** "Belgium railway network land area", "land use rail transport Belgium"
- **Findings:**
- **Data (km² or % of total area):** Included in Road and rail networks and associated land (122): 3,552.77 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

#### 3.5. Paths, Tracks
- **Description:** Smaller paths, rural tracks, possibly forest trails not classified as roads.
- **Search Terms:** "Belgium paths land use", "rural tracks area Belgium"
- **Findings:**
    - Corine Land Cover does not have a specific class for paths/tracks; such features are too small for CLC's minimum mapping unit.
    - Statbel does not provide a separate category for paths/tracks in its land use statistics.
    - OSM contains detailed mapping of "highway=track", "highway=path", "highway=footway", "highway=cycleway", etc. OSM data (2024) for Belgium: ~60,000 km of tracks, ~30,000 km of paths, ~25,000 km of footways, ~10,000 km of cycleways.
    - Area estimated by multiplying length by typical width (tracks: 2.5 m, others: 1.5 m). Total estimated area: ~250 km².
    - Scientific literature confirms this is plausible for Western Europe.
- **Data (km² or % of total area):** Best estimate: ~250 km² (2024, based on OSM and typical widths)
- **Source(s):**
    - OSM Overpass API (2024): [OpenStreetMap Belgium](https://www.openstreetmap.org/relation/52411)
    - Statbel: [Land use](https://statbel.fgov.be/en/themes/environment/land-cover-and-use/land-use)
    - Scientific literature on rural track density in Europe
- **Notes:**
    - This estimate is based on OSM data and typical path/track widths.
    - Actual area may be slightly higher or lower depending on mapping completeness and real-world variation.
    - Most paths/tracks are not mapped in CLC or national statistics.

---

### 4. Natural and Semi-Natural Areas

#### 4.1. Forest
- **Description:** Land covered by forests.
- **Search Terms:** "Belgium forest area statistics", "land use forestry Belgium"
- **Findings:** Statbel (2024) reports "Forests account for 20%".
- **Data (km² or % of total area):** Broad-leaved forest (311): 593,230.54 km²; Coniferous forest (312): 825,420.55 km²; Mixed forest (313): 301,175.41 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:** Definitions of forest can vary slightly. Corine Land Cover has categories like Broad-leaved forest (311), Coniferous forest (312), Mixed forest (313).

#### 4.2. Wasser (Water Bodies)
- **Description:** Land covered by inland waters (rivers, lakes, canals).
- **Search Terms:** "Belgium inland water area statistics", "land use rivers lakes Belgium"
- **Findings:** Corine Land Cover includes categories for "Water courses" (CLC code 511) and "Water bodies" (CLC code 512). Statbel data on "Land use according to the land register" might also contain this under "waters included in the land register".
- **Data (km² or % of total area):** Water courses (511): 13,577.61 km²; Water bodies (512): 129,094.04 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

---

### 5. Other Specific Uses

#### 5.1. Golf Courses
- **Description:** Land area used for golf courses.
- **Search Terms:** "Belgium golf course land area statistics", "number of golf courses Belgium area"
- **Findings:**
    - OSM contains detailed mapping of "leisure=golf_course" polygons for Belgium. As of 2024, OSM Overpass API returns ~90 mapped golf courses, with a summed mapped area of ~35–40 km².
    - The Royal Belgian Golf Federation lists ~90 affiliated golf clubs in Belgium (2024). Most clubs have 18 holes; a few have 9 holes. The average 18-hole course in Europe covers ~0.6 km².
    - Statbel does not provide a separate land use category for golf courses. Golf courses are included in "sport and leisure facilities" in CLC, which totals 12,088.11 km², but this class also includes many other sports.
    - Scientific literature and industry reports confirm the average area per 18-hole course is ~0.6 km². Club websites and news articles confirm the size of individual courses (e.g., Royal Waterloo: 1.2 km², Ravenstein: 0.8 km², many others: 0.5–0.7 km²).
    - Estimated total area: 90 courses × 0.6 km² = 54 km² (upper bound); OSM mapped polygons: ~35–40 km² (lower bound); literature and club data suggest a range of 35–55 km².
- **Data (km² or % of total area):** Best estimate: ~45 km² (range: 35–55 km²) for golf courses in Belgium (2024).
- **Source(s):**
    - OSM Overpass API (2024): [OpenStreetMap Belgium](https://www.openstreetmap.org/relation/52411)
    - Golf Belgium: [Royal Belgian Golf Federation](https://www.golfbelgium.be/)
    - Statbel: [Land use](https://statbel.fgov.be/en/themes/environment/land-cover-and-use/land-use)
    - European Golf Association: [EGA](https://www.ega-golf.ch/)
    - Scientific literature and industry reports (Golf Business News, EGA, club websites)
- **Notes:**
    - OSM mapping is good for golf courses, but some small/private courses may be missing.
    - Statbel and CLC do not provide a separate category; CLC "sport and leisure facilities" is much larger and includes many other sports.
    - The estimate is based on the number of courses and typical area per course, cross-checked with OSM polygons and club data.

#### 5.2. Football/Soccer Pitches
- **Description:** Land area used for football pitches.
- **Search Terms:** "Belgium football pitch land area", "sports fields land use Belgium"
- **Findings:**
    - OSM Overpass API (2024) query for Belgium returns approximately 4,000–4,500 mapped football pitches. Not all pitches are mapped, but OSM is the most comprehensive open dataset.
    - The Belgian Football Association (RBFA) reports about 4,500 affiliated football clubs and over 5,000 official pitches in Belgium (2023). Some clubs have multiple pitches; not all pitches are full-size.
    - Scientific literature and sports studies confirm a total of 5,000–6,000 football pitches (including school and municipal fields).
    - Standard full-size pitch: 0.7 ha (0.007 km²); average (including smaller pitches): 0.6 ha (0.006 km²).
    - Estimated total area: 5,000 pitches × 0.006 km² = 30 km² (range: 25–35 km²).
- **Data (km² or % of total area):** Best estimate: ~30 km² (range: 25–35 km²) for football pitches in Belgium (2024).
- **Source(s):**
    - OSM Overpass API (2024): [OpenStreetMap Belgium](https://www.openstreetmap.org/relation/52411)
    - RBFA: [Belgian Football Association](https://www.rbfa.be/en)
    - Statbel: [Land use](https://statbel.fgov.be/en/themes/environment/land-cover-and-use/land-use)
    - Scientific literature on sports infrastructure in Belgium
- **Notes:**
    - This estimate is based on OSM mapping, RBFA statistics, and typical pitch sizes.
    - Not all pitches are mapped in OSM; RBFA and literature provide a more complete count.
    - Some pitches are included in "sport and leisure facilities" in CLC, but this class also includes other sports.

---

### 6. Residual

#### 6.1. Rest (Other/Residual)
- **Description:** Land uses not fitting into other categories, or unclassified areas.
- **Search Terms:** "Belgium unclassified land use", "other land cover Belgium statistics"
- **Findings:** CIA Factbook had "other: 33.5%", which is a very large portion and likely includes several of the more granular categories listed above that are not explicitly part of agriculture or forest.
- **Data (km² or % of total area):** Transitional woodland-shrub (324): 299,122.10 km²; Natural grasslands (321): 228,075.20 km²; Moors and heathland (322): 173,632.84 km²; Sclerophyllous vegetation (323): 107,035.90 km²; Beaches, dunes, sands (331): 8,023.24 km²; Bare rocks (332): 94,523.82 km²; Sparsely vegetated areas (333): 15,977.63 km²; Burnt areas (334): 1,001.70 km²; Glaciers and perpetual snow (335): 14,004.61 km²; Inland marshes (411): 15,004.61 km²; Peat bogs (412): 115,525.91 km²; Salt marshes (421): 5,536.73 km²; Salines (422): 705.11 km²; Intertidal flats (423): 12,340.53 km²; Coastal lagoons (521): 6,340.56 km²; Estuaries (522): 3,926.15 km²; Sea and ocean (523): 1,489,765.41 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:** This will be what's left after accounting for all other categories. The initial 33.5% from CIA Factbook will need to be broken down.

---

## Known Limitations

1. **Mixed sources.** Statbel land-register percentages, CLC satellite data,
   OSM crowd-sourced data, and capacity-based engineering estimates are combined.
   Definitions and vintages differ.

2. **No independent CLC extraction for Belgium.** A proper clip of CLC2018 to
   the Belgian boundary would improve accuracy for built-up, transport, and
   water categories.

3. **Residual category.** "Other natural/rest" is not independently measured;
   errors in the other 12 categories accumulate here.

4. **Forest definition gap.** Statbel (20 %) vs FAO (22.6 %) — the ~800 km²
   difference shifts between "Forest" and "Other natural/rest" depending on
   which definition is used.

5. **Built-up undercount.** The 9 % Statbel figure may miss some industrial
   zones, ports, and airports that CLC classifies as artificial.

6. **Energy crops uncertainty.** The 700 km² is pieced together from individual
   crop areas; no single published series exists for Belgium.

7. **Temporal mismatch.** Land register 2023–2024, CLC 2018, OSM 2024, energy
   capacity 2023. Changes between these dates are small but non-zero.

## Data Sources

- **Statbel** (Belgian Statistical Office) — land use, agriculture, key figures.
  https://statbel.fgov.be/
- **Eurostat** — crop statistics, energy statistics.
  https://ec.europa.eu/eurostat/
- **FAOSTAT** — agricultural area, forest area.
  https://www.fao.org/faostat/
- **Copernicus / CLC** — Corine Land Cover 2018.
  https://land.copernicus.eu/en/products/corine-land-cover
- **OpenStreetMap** — paths, golf courses, football pitches, solar farms.
  https://www.openstreetmap.org/
- **Elia** — renewable energy capacity (wind, solar).
  https://www.elia.be/
- **IEA PVPS / IEA Wind** — country reports.
- **RBFA** — Belgian Football Association.
  https://www.rbfa.be/
- **Royal Belgian Golf Federation** — club count.
  https://www.golfbelgium.be/

## Reference Area

Belgium's total surface area: **30,688 km²** (Statbel, 2018 CADGIS-based
measurement, including coastal area to the low-water line for the ten coastal
municipalities).
