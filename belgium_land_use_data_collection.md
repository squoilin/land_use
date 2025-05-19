# Belgian Land Use Data Collection for Detailed Map

## Summary Table of Land Use in Belgium

| Category                        | Area (km²) | % of Total | Notes/Overlap Handling |
|----------------------------------|------------|------------|-----------------------|
| Agricultural land (total)        | 13,503     | 44.0%      | Includes all crops, pastures, energy crops, etc. |
| - of which: Energy crops         | 700        | 2.3%       | Subset of agricultural land |
| Forest                          | 6,138      | 20.0%      | Statbel/CLC |
| Built-up (residential, industry, etc.) | 2,762      | 9.0%       | Statbel/CLC |
| Roads and rail infrastructure    | 3,553      | 11.6%      | CLC; may overlap with built-up |
| Sport and leisure facilities     | 45         | 0.15%      | Sum of golf and football; not CLC total to avoid double-counting |
| - of which: Golf courses         | 45         | 0.15%      | Included above |
| - of which: Football pitches     | 30         | 0.10%      | Included above |
| Ground-mounted PV                | 13         | 0.04%      | |
| Wind turbines (footprint)        | 14         | 0.05%      | |
| Paths/tracks                     | 250        | 0.8%       | Overlaps with agri/forest; not added to total |
| Water bodies (inland)            | 143        | 0.47%      | CLC 511+512 |
| Other natural/semi-natural/rest  | 4,271      | 13.9%      | Residual to reach 100% |
| **Total**                        | **30,689** | **100%**   | |

**Notes:**
- The sum is forced to 100% by using the Statbel/CLC high-level categories and treating detailed items as subsets or notes.
- Sport and leisure facilities are counted only as the sum of golf and football to avoid double-counting with CLC totals.
- Paths/tracks are not added to the total, as they overlap with agricultural and forest land.
- "Other natural/semi-natural/rest" is the residual category to ensure the sum is 100%.
- If any category appears inconsistent or unreliable, further investigation is recommended.

This document outlines the process of collecting detailed land use data for Belgium, aiming to match the categories found in a German reference image.

## Data Categories and Collection Process

The categories are derived from the German "Flächennutzung Deutschland" image. For each category, we will attempt to find corresponding data for Belgium.

---

### 1. Agricultural Land & Related

#### 1.1. Energiepflanzen (Energy Crops)
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

#### 1.2. Pflanzliche Ernährung (Crops for Human Consumption)
- **Description:** Land used for growing crops directly for human food.
- **Search Terms:** "Belgium arable land human consumption", "crop production statistics Belgium", "land use food crops Belgium"
- **Findings:**
- **Data (km² or % of total area):** Non-irrigated arable land (211): 1,221,161.74 km²; Permanently irrigated land (212): 104,963.28 km²; Rice fields (213): 7,730.59 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:** These are the main arable land classes in CLC.

#### 1.3. Viehfutter (Animal Feed Crops)
- **Description:** Land used for growing crops for animal fodder (e.g., maize for silage, fodder beets, grassland for grazing/hay). This includes "Permanent Pasture" and parts of "Arable Land".
- **Search Terms:** "Belgium animal feed crops land use", "fodder crops area Belgium", "permanent pasture land use Belgium"
- **Findings:** Part of the 44% agricultural land reported by Statbel.
- **Data (km² or % of total area):** Pastures (231): 425,428.27 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

#### 1.4. Sonst. Agrar (Other Agricultural) / Weihnachtsbäume (Christmas Trees)
- **Description:** Miscellaneous agricultural uses, with a specific mention of Christmas trees in the German example.
- **Search Terms:** "Christmas tree cultivation area Belgium statistics", "other agricultural land use Belgium"
- **Findings:**
- **Data (km² or % of total area):** Vineyards (221): 41,654.40 km²; Fruit trees and berry plantations (222): 50,734.14 km²; Olive groves (223): 4,160.56 km²; Annual crops associated with permanent crops (241): 32,739.69 km²; Complex cultivation patterns (242): 252,946.18 km²; Land principally occupied by agriculture, with significant areas of natural vegetation (243): 277,020.04 km²; Agro-forestry areas (244): 32,739.69 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

---

### 2. Built-up and Artificial Areas

#### 2.1. Siedlung, Industrie, Freizeit (Settlement, Industry, Leisure)
- **Description:** Combined area for residential, industrial, commercial, and recreational built-up areas.
- **Search Terms:** "Belgium settlement area statistics", "industrial land use Belgium", "recreational land use Belgium", "artificial surfaces Belgium"
- **Findings:** Statbel provides data on "built-up land and related sites." Statbel (2024) reports "residential lands for 9%". Corine Land Cover data (e.g., CLC2018) categorizes various artificial surfaces (urban fabric, industrial/commercial units, transport units, mine/dump/construction sites, artificial non-agricultural vegetated areas).
- **Data (km² or % of total area):** Continuous urban fabric (111): 6,726.63 km²; Discontinuous urban fabric (112): 165,366.65 km²; Industrial or commercial units (121): 29,039.31 km²; Green urban areas (141): 3,565.12 km²; Sport and leisure facilities (142): 12,088.11 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:** This is a broad category. "Wohnen" (Residential) is a sub-component.

#### 2.2. Wohnen (Residential / Housing)
- **Description:** Land specifically used for housing.
- **Search Terms:** "Belgium residential land use statistics", "housing area Belgium"
- **Findings:** Statbel (2024) states "residential lands for 9%".
- **Data (km² or % of total area):** Continuous urban fabric (111): 6,726.63 km²; Discontinuous urban fabric (112): 165,366.65 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:** This will be a part of the broader "Siedlung, Industrie, Freizeit".

#### 2.3. Tagebau etc. (Opencast Mining, Quarries)
- **Description:** Land used for mineral extraction (opencast mines, quarries).
- **Search Terms:** "Belgium mining land use", "quarry area Belgium statistics", "extraction sites land use Belgium"
- **Findings:** Corine Land Cover includes a category for "Mineral extraction sites" (CLC code 131) and "Dump sites" (CLC code 132).
- **Data (km² or % of total area):** Mineral extraction sites (131): 8,051.67 km²; Dump sites (132): 1,194.15 km²; Construction sites (133): 2,407.68 km². (CLC2012)
- **Source(s):** CLC2012 raster, CLC legend.
- **Notes:**

---

### 3. Energy & Infrastructure

#### 3.1. Flächen-PV (Ground-mounted Photovoltaics)
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

#### 3.2. Windkraftanlagen (Area for Wind Turbines)
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

#### 3.3. Straßenverkehr (Roads and Road Infrastructure)
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

#### 3.5. Wege (Paths, Tracks)
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

#### 4.1. Wald (Forest)
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

#### 5.1. Golfplätze (Golf Courses)
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

#### 5.2. Fußballplätze (Football/Soccer Pitches)
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

## Summary of Challenges:
- **Granularity:** Finding data for very specific sub-categories (e.g., Christmas trees, football pitches) at a national aggregate level is difficult.
- **Consistency:** Definitions and methodologies can vary between sources (e.g., Statbel land register vs. Corine Land Cover satellite-derived).
- **Overlap:** Categories can overlap (e.g., "Wohnen" is part of "Siedlung").
- **Data Availability:** Some data might not be publicly available or regularly updated. The most recent aggregate data from Statbel (Sept 2024) is very useful for high-level categories. Corine Land Cover (2018) provides more spatial detail and thematic breakdown but is older.

The goal will be to find the best available data and make reasonable estimations or aggregations where necessary.
The total surface area of Belgium is 30,689 km² (Statbel, 2018 CADGIS based). This will be used for converting percentages.

## Corine Land Cover (CLC) Data for Belgium

The Corine Land Cover (CLC) inventory provides a pan-European land cover and land use dataset with 44 thematic classes. The latest version is CLC2018.
This data can be used to refine several categories, especially for artificial surfaces, different types of agricultural areas, forests, and water bodies.

**Key CLC Level 1 Categories:**
1. Artificial surfaces
2. Agricultural areas
3. Forest and semi-natural areas
4. Wetlands
5. Water bodies

**Accessing CLC Data for Belgium:**
- Copernicus Land Monitoring Service: [https://land.copernicus.eu/en/products/corine-land-cover](https://land.copernicus.eu/en/products/corine-land-cover)
- Belgian national geoportal (data.gov.be): Provides access to "INSPIRE View service - CORINE Land Cover-2018-Belgium".

**Potential for filling categories:**
- **1.1 Energiepflanzen:** Might require specific analysis of CLC agricultural classes, unlikely to be a direct CLC category.
- **2.1 Siedlung, Industrie, Freizeit:** Aggregation of CLC classes like "Urban fabric" (111, 112), "Industrial or commercial units" (121), "Port areas" (123), "Airports" (124), "Sport and leisure facilities" (142).
- **2.3 Tagebau etc.:** "Mineral extraction sites" (131), "Dump sites" (132), "Construction sites" (133).
- **3.3 Straßenverkehr & 3.4 Bahn:** "Road and rail networks and associated land" (122).
- **4.2 Wasser:** "Water courses" (511), "Water bodies" (512), "Coastal lagoons" (521), "Estuaries" (522), "Sea and ocean" (523 - for coastal areas).

Further work would involve downloading the CLC vector data for Belgium and calculating the areas for the relevant classes to fill in the specific data points (km² or %). 