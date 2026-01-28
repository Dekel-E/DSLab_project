# Hotel Reality Gap Analysis - Data Science Project

## Project Overview

This project analyzes the gap between hotel expectations (what hotels claim/promise) and reality (what guests actually experience). Using data from Booking.com, OpenStreetMap, Google Maps, and crime statistics, we build a machine learning model to predict actual hotel quality and identify "overrated" properties.

**Key Innovation**: By combining spatial analysis (nearby POIs, noise sources, transit) with NLP sentiment analysis of guest reviews, we detect hotels that may disappoint guests despite high ratings.

---

## Dataset

The analysis covers **62,643 hotels** across **10 major cities**:
- Amsterdam, Bangkok, Dubai, Eilat, Haifa, London, New York, Rome, Tel Aviv, Tokyo

**Data Sources**:
- **Booking.com**: Hotel listings, reviews, amenities, scores
- **OpenStreetMap (OSM)**: Points of interest (restaurants, nightlife, transport, parks)
- **Google Maps**: Additional POI data with ratings and reviews
- **Numbeo**: City-level crime statistics

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
1. **Data Loading**
   - Reads hotel data from Azure Blob Storage (parquet format)
   - Filters to 10 target cities
   - Standardizes city names (handles variations like "NYC" → "New York")

2. **Geographic Validation**
   - Calculates distance from city center using Haversine formula
   - Removes outliers (hotels too far from city center)
   - Validates coordinates

3. **Feature Extraction**
   - **Amenities**: Extracts flags for WiFi, parking, AC, pool, gym, etc. (18 amenity types)
   - **House Rules**: Check-in/out times, pet policies, smoking rules, quiet hours
   - **Property Surroundings**: Number and distance of nearby POIs
   - **Claims Analysis**: Detects keywords in descriptions (claims_quiet, claims_central, claims_luxury, etc.)
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
1. **Data Loading**
   - Loads POI data from 6 cities (13,660 POIs)
   - Richer data: ratings, reviews, categories

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
1. **Geospatial Indexing**
   - Adds GeoHash to hotels and POIs for efficient spatial queries
   - Creates spatial indices for fast distance calculations

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

#### Part B: NLP Analysis
1. **Review Text Processing**
   - Loads review text from all hotels
   - Cleans and normalizes text (lowercase, remove special chars)

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

4. **Sentiment Analysis**
   - Uses PySpark ML or TextBlob for sentiment scoring
   - Calculates **sentiment polarity** (-1 to +1)
   - Calculates **sentiment intensity** (strength of emotion)
   - Creates **complaint ratio** (complaints / total sentiment words)

5. **Gap Indicators Created**
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
1. **Random Forest Model**
   - 100 trees, max depth 10
   - **Performance**:
     - Test RMSE: 1.179
     - Test R²: 0.723
     - Test MAE: 0.792

2. **Gradient Boosted Trees (GBT)**
   - 50 iterations, max depth 10
   - **Performance** (Better!):
     - Test RMSE: 0.814
     - Test R²: 0.867
     - Test MAE: 0.484

3. **Selected Model**: GBT (better performance)

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
**Top 5 Drivers of Hotel Scores**:
1. `number_of_reviews` (35.7%) - Social proof
2. Spatial features (30.6%) - Location quality
3. Expectation features (17.4%) - What hotel claims
4. NLP features (8.6%) - Guest complaints/praise
5. Crime features (7.7%) - City safety

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
Place the following files in your Databricks Workspace:
```
/Workspace/Users/[your-email]/
├── booking_1_9.parquet (Booking.com data)
├── amsterdam.csv (OSM)
├── bangkok.csv (OSM)
├── dubai.csv (OSM)
├── eilat.csv (OSM)
├── haifa.csv (OSM)
├── london.csv (OSM)
├── new_york_city.csv (OSM)
├── rome.csv (OSM)
├── tel_aviv.csv (OSM)
├── tokyo.csv (OSM)
├── amsterdam_final.csv (Google Maps)
├── bangkok_final.csv (Google Maps)
├── dubai_final.csv (Google Maps)
├── eilat_final.csv (Google Maps)
├── rome_final.csv (Google Maps)
├── tokyo_final.csv (Google Maps)
├── crime_amsterdam_numbeo.csv
├── crime_bangkok_numbeo.csv
├── crime_dubai_numbeo.csv
├── crime_eilat_numbeo.csv
├── crime_haifa_numbeo.csv
├── crime_london_numbeo.csv
├── crime_new_york_numbeo.csv
├── crime_rome_numbeo.csv
├── crime_tel_aviv_numbeo.csv
└── crime_tokyo_numbeo.csv
```

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
- **RMSE**: 0.814 (predicts within ±0.8 points on 10-point scale)
- **R² Score**: 0.867 (explains 86.7% of variance)
- **MAE**: 0.484 (average error of 0.48 points)

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


