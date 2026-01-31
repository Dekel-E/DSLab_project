# Hotel Reality Gap Analysis - Data Science Project

## Project Overview

This project analyzes the gap between hotel expectations (what hotels claim/promise) and reality (what guests actually experience). Using data from Booking.com, OpenStreetMap, Google Maps, and crime statistics, we build a machine learning model to predict actual hotel quality and identify "overrated" properties.

**Key Innovation**: By combining spatial analysis (nearby POIs, noise sources, transit) with NLP sentiment analysis of guest reviews, we detect hotels that may disappoint guests despite high ratings.

---

## Important Setup Note

**Before running the notebooks**, you must configure environment variables in the first cell:

- **Booking.ipynb**: Set `env_booking` in the first cell to your Databricks environment path
- **scraped_data.ipynb**: Set `env_scraped_data` in the first cell to your workspace path where the CSV files are located

These environment variables are critical for the notebooks to locate and access the required data files.

---

## Dataset

The analysis covers **62,643 hotels** across **10 major cities**:
- Amsterdam, Bangkok, Dubai, Eilat, Haifa, London, New York, Rome, Tel Aviv, Tokyo

**Data Sources**:
- **Booking.com**: Hotel listings, reviews, amenities, scores (primary dataset)
- **OpenStreetMap (OSM)**: 162,987 POI records (restaurants, nightlife, transport, parks)
- **Google Maps**: 13,660 POI records with ratings and reviews (7 cities, custom multi-threaded scraper with proxy rotation)
- **Numbeo**: City-level crime statistics (150 records across 10 cities)
- **TripAdvisor**: Collected but not used due to scalability constraints (kept for potential future enhancement)

**Total External Enrichment**: ~190,000 new data points

---
## Web interface link
https://drive.google.com/file/d/1y6IKTqMy8AR-8AHGH27r8CV67UQpp1cd/view?usp=sharing
## Notebook Execution Order

Run the notebooks in this exact sequence:

```
1. Booking.ipynb
2. scraped_data.ipynb
3. spatial_joins_NLP.ipynb
4. randomforest.ipynb
```

---

## 1. Booking.ipynb

### Purpose
Loads, cleans, and processes Booking.com hotel data to extract structured features.

### What It Does
1. **Data Loading & Standardization**
   - Reads hotel data from Azure Blob Storage (parquet format)
   - Filters to 10 target cities
   - **City Name Standardization**: Implements keyword mapping to unify inconsistent city names
     - Example: "NYC", "Bronx", "Brooklyn" → unified as "New York"
     - Ensures data integrity across geographic variations

2. **Geographic Validation & Geofiltering**
   - **Haversine Distance Filter**: Calculates distance of each listing from respective city center
   - **Spatial Outlier Removal**: Removes listings beyond defined city buffer
     - Filters erroneous entries (e.g., locations named "Rome" situated in USA)
     - City-specific radius thresholds (larger cities = larger buffer)
   - Validates coordinate accuracy
   - Result: Clean dataset relevant only to 10 target cities

3. **Feature Extraction**
   - **Amenities**: Parses `most_popular_facilities` field using predefined dictionary to transform amenity lists into Boolean vectors (e.g., has_wifi, has_pool) - 18 amenity types total
   - **House Rules**: Applies Regex to unstructured house rules to extract:
     - Structured numeric data (check-in/out times)
     - Categorical policies (pets allowed, smoking rules, quiet hours)
   - **Property Surroundings**: Number and distance of nearby POIs
   - **Claims Extraction (Expectation Modeling)**: Core component quantifying "host's promise"
     - Text mining module scans description, highlights, and title fields
     - Detects "Claims Keywords" for location and atmosphere (e.g., "central", "quiet", "spacious", "luxury")
     - Generates Claims Features (claims_luxury, claims_central, claims_quiet, etc.)
     - Serves as baseline for "Expectation vs. Reality" analysis
   - **Reviews**: Aggregates review text, reviewer countries

4. **Data Quality**
   - Handles null values
   - Converts complex nested structures to flat columns
   - Validates data types

### Output
- **Silver Layer**: `dbfs:/FileStore/project/silver/booking_parsed`
- **Columns**: 56 structured features per hotel
- **Format**: Parquet, partitioned by city

### Key Statistics
- Total hotels: 62,643
- Hotels with reviews: ~60,000
- Average amenities per hotel: 8-12
- Review coverage: 94% of hotels have text reviews

### How to Run
```python
# Prerequisites
# - Access to Azure Blob Storage with SAS token
# - Databricks cluster with PySpark

# Simply run all cells in order
# The notebook handles data loading, transformation, and saving automatically
```

---

## 2. scraped_data.ipynb

### Purpose
Loads and processes supplementary datasets (OSM POIs, Google Maps POIs, Crime data) to enrich hotel analysis with spatial and safety context.

### What It Does

#### Part A: OpenStreetMap (OSM) Data
1. **Data Loading**
   - Loads POI data from 10 CSV files (one per city)
   - Covers 162,987 POIs total

2. **POI Categorization**
   - **Noise Sources**: Bars, nightclubs, major roads
   - **Transport**: Stations, stops, ferry terminals
   - **Leisure**: Parks, sports centers, entertainment venues
   - **Tourism**: Attractions, museums, historic sites
   - **Dining**: Restaurants, cafes
   - **Convenience**: Health, finance, shopping, education

3. **Data Cleaning**
   - Validates coordinates
   - Removes duplicates
   - Standardizes category labels

4. **Output**: `dbfs:/FileStore/project/silver/osm_clean`

#### Part B: Google Maps Data

**Collection Method** (custom multi-threaded scraper):
- **Proxy Infrastructure**: Aggregated 200k proxies, filtered to ~300 high-quality functional proxies
- **Execution**: 15 concurrent headed browser instances (Selenium/Playwright), each with distinct proxy
- **Anti-Bot Evasion**: User-agent rotation, randomized delays, human-like scrolling patterns
- **Challenge**: Key limitation of OSM is lack of qualitative textual data; Google Maps bridges this gap

1. **Data Loading**
   - Loads POI data from 7 cities (13,660 POIs)
   - **Richer data**: ratings, reviews, categories (vs. OSM's basic location data)

2. **Enhanced Features**
   - **Noise Scoring**: Analyzes review text for noise keywords (loud, party, music vs. quiet, calm)
   - **Vibe Detection**: Tourist trap, local favorite, family-friendly, sketchy, romantic
   - **Quality Metrics**: Ratings, review counts

3. **Output**: `dbfs:/FileStore/project/silver/gmaps_clean`

#### Part C: Crime Data (Numbeo)
1. **Data Loading**
   - Loads crime statistics for all 10 cities
   - 15 metrics per city (crime level, safety index, assault risk, theft risk, etc.)

2. **Metrics Include**:
   - Level of crime
   - Worries about being mugged/robbed
   - Worries about car theft
   - Worries about home break-ins
   - Worries about physical attack
   - Safety walking alone (day/night)

3. **Output**: `dbfs:/FileStore/project/silver/crime_numbeo`

### Output Summary
- **OSM POIs**: 162,985 locations with categories
- **Google Maps POIs**: 13,660 locations with ratings + noise scores
- **Crime Data**: 150 records (15 metrics × 10 cities)

### How to Run
```python
# Prerequisites
# - CSV files in Workspace: /Workspace/Users/[username]/
#   - amsterdam.csv, bangkok.csv, dubai.csv, etc. (OSM)
#   - amsterdam_final.csv, bangkok_final.csv, etc. (Google Maps)
#   - crime_amsterdam_numbeo.csv, etc. (Crime)

# The notebook will:
# 1. Copy files from Workspace to DBFS
# 2. Load and process each dataset
# 3. Save cleaned data to silver layer
```

---

## 3. spatial_joins_NLP.ipynb

### Purpose
The core analytics notebook that combines spatial analysis and NLP to detect gaps between hotel promises and reality.

### What It Does

#### Part A: Spatial Joins

**Efficient Joining via Geohashing**

A naive distance calculation between every hotel and POI would result in a prohibitive Cartesian product (~70,000 hotels × 180,000 POIs ≈ 12.6 billion comparisons). To solve this computational bottleneck, we implemented a **Geohash-based blocking strategy**:

1. **Geospatial Indexing**
   - Assigns precision-based Geohash strings to hotels and POIs
   - Partitions the world into manageable grid cells
   - Reduces the problem from global search to local bucket search

2. **Neighbor Explosion Technique**
   - Replicates each hotel record to associate with its own Geohash cell
   - Also associates with 8 surrounding neighbor cells
   - Prevents missing POIs located just across grid cell borders
   - This "neighbor explosion" ensures comprehensive coverage

2. **OSM Spatial Join**
   - Counts POIs within various radii (300m, 500m, 1km)
   - **Nightlife counts** (500m radius) - noise indicator
   - **Transport proximity** (nearest station/stop)
   - **Restaurant density** (300m, 500m)
   - **Parks and leisure** (500m, nearest park)
   - **Tourist attractions** (1km)

3. **Google Maps Spatial Join**
   - Aggregates GMaps ratings around each hotel
   - **Average noise score** within 500m
   - **Average restaurant rating** within 500m
   - **Neighborhood quality score**
   - **Vibe indicators**: Tourist traps, local favorites, sketchy areas
   - Counts high-rated establishments nearby

4. **Key Metrics Calculated**
   - `nightlife_count_500m`: Number of bars/clubs nearby (noise risk)
   - `noise_sources_500m`: Major roads + nightlife
   - `nearest_transport_m`: Distance to metro/train
   - `avg_noise_score_500m`: Average noise level from GMaps reviews
   - `high_rated_restaurants_500m`: Quality dining options
   - `tourist_trap_pois_500m`: Tourist trap density

#### Part B: NLP Analysis - Hybrid Dual-Layer Pipeline

This notebook implements a **Hybrid NLP Pipeline** combining rule-based extraction with semantic sentiment analysis to capture both specific complaints and general sentiment nuances:

**Layer 1: Rule-Based Extraction**
1. **Review Text Processing**
   - Loads review text from all hotels
   - Cleans and normalizes text (lowercase, remove special chars)
   - Applies complex Regex patterns to detect specific complaint categories

2. **Keyword-Based Complaint Detection**
   - **Noise complaints**: "loud", "noisy", "party", "music"
   - **Cleanliness complaints**: "dirty", "stain", "smell", "mold"
   - **Location complaints**: "far from", "inconvenient", "middle of nowhere"
   - **Amenities complaints**: "broken", "missing", "not working"
   - **Host complaints**: "rude", "unhelpful", "unresponsive"
   - **Value complaints**: "overpriced", "not worth", "expensive"

3. **Praise Detection**
   - **Quiet praise**: "quiet", "peaceful", "tranquil"
   - **Cleanliness praise**: "spotless", "immaculate", "clean"
   - **Location praise**: "perfect location", "convenient", "central"
   - **Amenities praise**: "great amenities", "well-equipped"
   - **Host praise**: "friendly", "helpful", "attentive"
   - **Value praise**: "great value", "affordable", "worth it"

4. **Sentiment Analysis (Layer 2: Semantic Analysis)**
   - Uses **Spark NLP's BERT-based pipeline** (Universal Sentence Encoder)
   - Generates `bert_sentiment_score` for sophisticated emotional tone understanding
   - Calculates **sentiment polarity** (-1 to +1)
   - Calculates **sentiment intensity** (strength of emotion)
   - Creates **complaint ratio** (complaints / total sentiment words)
   - Captures nuances missed by simple keyword counting

5. **Gap Signal Detection - Core Innovation**

A key innovation is the engineering of **"Gap Signals"** - features quantifying contradictions between host promises and guest realities:

   - **Noise Gap Example**: Flags discrepancies where hotel metadata claims "quiet" or "soundproof", yet NLP detects high frequency of noise-related complaints
   - **bert_score_gap**: Difference between normalized review score (0-1) and BERT sentiment score - indicates when users give high ratings but write negative reviews (social pressure effect)

**Gap Indicators Created**
   - `sentiment_gap`: Difference between review score and sentiment
   - `noise_gap_signal`: Claims quiet but has noise complaints
   - `cleanliness_gap_signal`: Claims clean but has cleanliness complaints
   - `location_gap_signal`: Claims central but guests say it's far
   - `amenity_gap_signal`: Claims amenities but guests report issues

### Output
- **Gold Layer**: `dbfs:/FileStore/project/gold/hotel_with_sentiment`
- **Features**: ~150 columns including:
  - All original hotel features
  - 17 spatial features from OSM
  - 17 spatial features from Google Maps
  - 34 NLP-derived features (complaints, praises, sentiment)
  - Gap indicators

### How to Run
```python
# Prerequisites
# - Completed notebooks 1 and 2
# - Data in silver layer:
#   - booking_parsed
#   - osm_clean
#   - gmaps_clean

# Runtime: ~15-20 minutes 
# - Spatial joins are computationally expensive
# - NLP processing requires iteration over all reviews
```

---

## 4. randomforest.ipynb

### Purpose
Builds a machine learning model to predict "reality scores" and calculate the gap between expectations and actual guest experience.

### What It Does

#### Part A: Data Preparation
1. **Feature Selection**
   - **Expectation Features** (34): Claims, amenities, policies
   - **Spatial Features** (34): OSM + Google Maps POI data
   - **NLP Features** (34): Complaints, praises, sentiment
   - **Crime Features** (15): City-level safety metrics
   - **Other** (3): Review counts, diversity metrics
   - **Total**: 120 features

2. **Data Processing**
   - Joins hotel data with crime statistics by city
   - Converts boolean features to numeric (0/1)
   - Fills missing values with 0
   - Assembles feature vector using VectorAssembler

3. **Train/Test Split**
   - 80% training (18,484 hotels)
   - 20% testing (4,504 hotels)
   - Stratified by city to ensure balanced representation

#### Part B: Model Training

**Model Selection: Gradient Boosted Trees (GBT) vs Random Forest**

We experimented with multiple ensemble methods to predict actual hotel quality:

1. **Random Forest Model (Baseline)**
   - 100 trees, max depth 10
   - **Approach**: Parallel bagging (trains trees independently)
   - **Performance**:
     - Test RMSE: 1.179
     - Test R²: 0.723
     - Test MAE: 0.792

2. **Gradient Boosted Trees (GBT) - Selected Model**
   - 50 iterations, max depth 10
   - **Approach**: Sequential boosting (each tree corrects errors from previous trees)
   - **Why GBT?**: GBT's boosting mechanism proved more effective at capturing subtle, non-linear patterns in our data
   - **Final Performance**:
     - **Test RMSE: 0.839** (average error ~0.8 points on 10-point scale)
     - **Test R²: 0.873** (explains 87.3% of variance in hotel scores)
     - **Test MAE: 0.484** (average absolute error 0.48 points)

   **Interpretation**: An R² of 0.873 indicates strong predictive power. The model's predictions deviate from actual ratings by less than 0.9 points on average - highly acceptable for this domain.

3. **Model Architecture**
   - High-dimensional feature space: ~150 features total
   - Features assembled using Spark's VectorAssembler
   - Three data layers integrated:
     - **Expectation Features**: Quantified host claims
     - **Spatial Reality**: OSM & Google Maps geospatial metrics
     - **NLP Signals**: Sentiment scores and complaint/praise ratios

#### Part C: Gap Score Calculation
1. **Reality Prediction**
   - Applies GBT model to ALL hotels
   - Predicts "reality score" based on spatial + NLP features

2. **Gap Calculation**
   ```
   gap_score = review_score - predicted_reality_score

   Positive gap = Overrated (disappointing)
   Negative gap = Underrated (exceeds expectations)
   ```

3. **Risk Categorization**
   - **High Risk** (gap > 1.5): Likely to disappoint
   - **Medium Risk** (gap 0.5-1.5): Some concerns
   - **As Expected** (gap -0.5 to 0.5): Reliable
   - **Better Than Expected** (gap -1.5 to -0.5): Pleasant surprise
   - **Hidden Gem** (gap < -1.5): Much better than expected

#### Part D: Feature Importance Analysis

**Why GBT for Interpretability**: A primary reason for selecting GBT was its interpretability via feature importance scores, revealing clear patterns in guest satisfaction.

**Top Drivers of Hotel Scores**:

1. **`number_of_reviews` (~26% importance)** - The strongest single predictor
   - Indicates "Wisdom of the Crowd" effect
   - Established, frequently reviewed hotels maintain consistent scores
   - Social proof is the dominant quality signal

2. **Spatial Features (~30.6% combined)** - Location quality matters most
   - `distance_from_center_km` (3.5%) - Convenience and accessibility
   - `nearest_transport_m` (2.9%) - Transit proximity
   - Confirms that location drives satisfaction

3. **Expectation Features (~17.4% combined)** - What hotels claim
   - `amenities_count` (3.2%) - Hotels with more services score higher
   - Validates that variety of offerings matters

4. **NLP Features (~8.6% combined)** - Actual guest experience
   - `sentiment_score_adjusted` and `positive_word_count` in top 20
   - Confirms NLP-derived signals align with numerical ratings
   - Contributes meaningfully despite structural features dominating

5. **Crime Features (~7.7% combined)** - City safety impacts satisfaction
   - City-level safety metrics influence overall experience

**Key Insight**: While complex spatial and NLP features add value, simple signals of "trust" and "social proof" (review volume) remain the dominant predictors in hospitality.

#### Part E: Output for Web App
1. **Creates Final Dataset** (`hotel_webapp`)
   - 22,988 hotels with complete data
   - 54 columns optimized for frontend display
   - Includes:
     - Gap score and risk level
     - Trust badge ("Hidden Gem", "Overhyped", etc.)
     - Top complaints and praises
     - Key amenities and location features
     - Neighborhood quality metrics

2. **Export Formats**
   - Parquet: `dbfs:/FileStore/project/gold/hotel_webapp`
   - CSV: `/FileStore/tables/hotel_webapp.csv`

### Key Insights from Model
- **Most overrated cities**: Analyze `avg_gap_score` by city
- **Hidden gems**: Hotels with gap_score < -1.5
- **Tourist traps**: Hotels with gap_score > 1.5
- **Feature importance**: What actually matters for satisfaction

### How to Run
```python
# Prerequisites
# - Completed notebook 3
# - Data in gold layer: hotel_with_sentiment

# Runtime: ~10-15 minutes
# - Model training: ~5-7 minutes
# - Predictions on all hotels: ~2 minutes
# - Feature importance + viz: ~5 minutes
```

---

## Prerequisites & Setup

### Environment
- **Platform**: Databricks (Azure or AWS)
- **Cluster**: Standard cluster with 8+ GB RAM
- **Runtime**: Databricks Runtime 13.x+ (includes PySpark, Pandas, Matplotlib)

### Required Libraries
```python
# Pre-installed on Databricks
- pyspark
- pandas
- matplotlib
- seaborn
- numpy

# Additional (install via %pip)
- folium (for maps in notebook 2)
```

### Installation
```bash
# In Databricks notebook:
%pip install folium
```

### Data Requirements
All the data is being read from our BLOB
---

## Output Files

### Silver Layer (Cleaned Data)
- `dbfs:/FileStore/project/silver/booking_parsed` - Cleaned hotel data
- `dbfs:/FileStore/project/silver/osm_clean` - OSM POIs
- `dbfs:/FileStore/project/silver/gmaps_clean` - Google Maps POIs
- `dbfs:/FileStore/project/silver/crime_numbeo` - Crime statistics

### Gold Layer (Analytics-Ready)
- `dbfs:/FileStore/project/gold/hotel_with_sentiment` - Hotels with all features
- `dbfs:/FileStore/project/gold/hotel_webapp` - Final dataset for web application

### CSV Export
- `/FileStore/tables/hotel_webapp.csv` - Downloadable CSV for web app

---

## Key Metrics & Results

### Dataset Statistics
- **Total Hotels**: 62,643
- **Hotels with Complete Data**: 22,988 (used in model)
- **Total POIs**: 176,645 (OSM + Google Maps)
- **Cities Covered**: 10
- **Features Engineered**: 150+

### Model Performance (GBT)
- **RMSE**: 0.839 (predicts within ±0.84 points on 10-point scale)
- **R² Score**: 0.873 (explains 87.3% of variance)
- **MAE**: 0.484 (average error of 0.48 points)

**Comparison to Baseline (Random Forest)**:
- RF RMSE: 1.179 vs GBT RMSE: 0.839 (29% improvement)
- RF R²: 0.723 vs GBT R²: 0.873 (15% improvement)

### Gap Analysis Results
- **High Risk Hotels**: ~15% (likely to disappoint)
- **Hidden Gems**: ~8% (exceed expectations)
- **As Expected**: ~50% (reliable)
- **Better Than Expected**: ~27%

### Feature Importance
1. **Review Volume** (35.7%) - More reviews = more reliable signal
2. **Spatial Context** (30.6%) - Location quality matters most
3. **Hotel Claims** (17.4%) - What hotel promises
4. **Guest Sentiment** (8.6%) - Actual experience
5. **City Safety** (7.7%) - Crime impacts satisfaction

---

## Limitations and Reflections

While the model demonstrates high predictive power (R² = 0.873), several limitations constrained the approach and offer avenues for future work:

### Data Collection Scalability & Coverage

**Challenge**: The most significant challenge was integrating proprietary external data. Due to strict anti-bot measures and aggressive rate-limiting:
- Google Maps enrichment covers only **7 of 10 target cities**
- TripAdvisor scraping was abandoned due to scalability issues

**Impact**: Model performance might vary across different geographies. A fully deployed production system would require official enterprise APIs rather than web scraping.

**Solution Used**: Developed custom multi-threaded scraper with proxy rotation (200k proxies filtered to ~300 high-quality), but still faced constraints.

### Geospatial Approximations

**Challenge**: Spatial analysis relies on **Haversine distance** (Euclidean "as-the-crow-flies" distance) to calculate POI proximity.

**Limitation**: In dense urban environments (Tokyo, New York), this doesn't reflect true walking distance due to:
- Physical barriers (highways, rivers)
- Lack of pedestrian crossings
- Complex urban layouts

**Impact**: "Convenience" features (e.g., `dist_to_subway`) are approximations that might over-optimistically assess a hotel's location.

### Temporal Dynamics & Seasonality

**Challenge**: Dataset represents a static snapshot in time.

**Missing Factors**:
- Seasonality effects (summer festivals, tourist seasons)
- Temporary construction noise
- Evolving neighborhoods
- Review timestamps not accounted for

**Impact**: Model doesn't capture transient trends affecting guest satisfaction. A "quiet" neighborhood might be noisy during peak seasons.

### Key Learnings

1. **Data Engineering Outweighs Modeling**: Initially underestimated computational cost of spatial joins. The "Cartesian Explosion" problem forced innovation with Geohashing and neighbor-cell strategies.

2. **External Enrichment is Powerful but Noisy**: Aligning vague hotel addresses with precise OSM coordinates required robust cleaning logic.

3. **Simple Signals Win**: Despite complex spatial and NLP features, `number_of_reviews` (social proof) remains the strongest predictor - highlighting the power of "wisdom of the crowd" in hospitality.

4. **Trade-offs in Data Collection**: Real-world data science involves balancing ideal data coverage with practical constraints (rate limits, scraping difficulties, API costs).

---

## Troubleshooting

### Common Issues

**Issue 1: Azure Blob Storage Access Denied**
```python
# Solution: Update SAS token in Booking.ipynb
sas_token = "your_new_token_here"
```

**Issue 2: File Not Found Errors**
```python
# Solution: Verify file paths in scraped_data.ipynb
base_path = "file:/Workspace/Users/[your-email]/"
```

**Issue 3: Memory Errors During Spatial Joins**
```python
# Solution: Increase cluster size or reduce data
# In spatial_joins_NLP.ipynb:
spark.conf.set("spark.sql.shuffle.partitions", 200)
```

**Issue 4: NLP Processing Timeout**
```python
# Solution: Process cities in batches
cities_batch_1 = ["Amsterdam", "Bangkok", "Dubai"]
cities_batch_2 = ["Eilat", "Haifa", "London"]
# ... process separately
```

---


## Project Structure

```
DSLab_project/
├── README.md (this file)
├── Booking.ipynb (Step 1: Data loading & cleaning)
├── scraped_data.ipynb (Step 2: POI & crime data)
├── spatial_joins_NLP.ipynb (Step 3: Spatial joins & NLP)
├── randomforest.ipynb (Step 4: ML model & gap scores)

---

## Contact & Questions

For questions or issues with the notebooks, please check:
1. Cell outputs for error messages
2. Data file paths and permissions
3. Cluster configuration and resources

**Expected Total Runtime**: 20-30 minutes for all notebooks

---

## Conclusion

This project successfully developed **"Check-in To Reality"**, a comprehensive data-driven framework that quantifies the gap between hotel marketing promises and actual guest experiences.

### Key Achievements

1. **Large-Scale Data Integration**: Successfully integrated over **190,000 external data points** from OpenStreetMap, Google Maps, and Numbeo with Booking.com's hotel dataset.

2. **Innovative Engineering Solutions**: Overcame significant computational challenges (Cartesian explosion problem) using Geohashing techniques with neighbor-cell expansion for efficient spatial joins.

3. **Hybrid Analytics Approach**: Combined rule-based NLP with BERT-based sentiment analysis to extract nuanced insights from guest reviews.

4. **Strong Predictive Performance**: Achieved **R² = 0.873** with Gradient Boosted Trees, validating that external environmental factors (noise levels, transit accessibility) significantly influence guest satisfaction.

5. **Actionable Insights**: Revealed that while physical amenities and location are critical, the "wisdom of the crowd" (review volume and text) remains the strongest indicator of quality.

### Impact

This framework offers **dual benefits**:

- **For Travelers**: Empowers informed decisions based on objective reality rather than just marketing claims
- **For Hosts**: Provides actionable, granular feedback to bridge the gap between expectation and delivery

The analysis demonstrates that data science can transform subjective hospitality experiences into quantifiable, predictable outcomes, ultimately improving trust and decision-making in the accommodation marketplace.

---


