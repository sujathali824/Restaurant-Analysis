"""
Restaurant Data Analysis Project
Complete solution for Cognifyz Level 1, 2, and 3 tasks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
df = pd.read_csv('Dataset .csv')

print("=" * 80)
print("RESTAURANT DATA ANALYSIS PROJECT")
print("=" * 80)

# ============================================================================
# LEVEL 1 - TASK 1: DATA EXPLORATION AND PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 1 - TASK 1: DATA EXPLORATION AND PREPROCESSING")
print("=" * 80)

# Basic dataset info
print("\n1. DATASET OVERVIEW:")
print(f"   Number of rows: {df.shape[0]}")
print(f"   Number of columns: {df.shape[1]}")
print(f"\n   Column names:\n   {', '.join(df.columns.tolist())}")

# Check for missing values
print("\n2. MISSING VALUES ANALYSIS:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_percent
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
else:
    print("   ✓ No missing values found in the dataset!")

# Handle missing values (if any)
# For Cuisines, we'll fill with 'Unknown'
if df['Cuisines'].isnull().sum() > 0:
    df['Cuisines'].fillna('Unknown', inplace=True)
    print(f"\n   ✓ Filled {missing['Cuisines']} missing Cuisines values with 'Unknown'")

# Data types
print("\n3. DATA TYPES:")
print(df.dtypes)

# Target variable analysis
print("\n4. TARGET VARIABLE ANALYSIS (Aggregate rating):")
rating_stats = df['Aggregate rating'].describe()
print(rating_stats)

# Distribution of ratings
rating_bins = [0, 1, 2, 3, 4, 5]
rating_labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
df['Rating_Range'] = pd.cut(df['Aggregate rating'], bins=rating_bins, labels=rating_labels, include_lowest=True)

print("\n   Rating Distribution:")
print(df['Rating_Range'].value_counts().sort_index())

# Class imbalance check
print("\n5. CLASS IMBALANCE ANALYSIS:")
print("\n   Rating Color Distribution:")
print(df['Rating color'].value_counts())
print("\n   Rating Text Distribution:")
print(df['Rating text'].value_counts())

# ============================================================================
# LEVEL 1 - TASK 2: DESCRIPTIVE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 1 - TASK 2: DESCRIPTIVE ANALYSIS")
print("=" * 80)

# Statistical measures for numerical columns
print("\n1. STATISTICAL MEASURES FOR NUMERICAL COLUMNS:")
numerical_cols = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
print(df[numerical_cols].describe())

# Categorical variables analysis
print("\n2. CATEGORICAL VARIABLES ANALYSIS:")

# Country distribution
print("\n   Country Code Distribution:")
country_dist = df['Country Code'].value_counts().head(10)
print(country_dist)

# City distribution
print("\n   Top 10 Cities by Number of Restaurants:")
city_dist = df['City'].value_counts().head(10)
print(city_dist)

# Top cuisines
print("\n3. TOP CUISINES:")
# Split cuisines (some restaurants have multiple cuisines)
all_cuisines = df['Cuisines'].str.split(',').explode().str.strip()
top_cuisines = all_cuisines.value_counts().head(10)
print(top_cuisines)

# ============================================================================
# LEVEL 1 - TASK 3: GEOSPATIAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 1 - TASK 3: GEOSPATIAL ANALYSIS")
print("=" * 80)

print("\n1. RESTAURANT DISTRIBUTION BY CITY:")
city_counts = df['City'].value_counts().head(10)
print(city_counts)

print("\n2. RESTAURANT DISTRIBUTION BY COUNTRY:")
country_counts = df['Country Code'].value_counts()
print(country_counts)

print("\n3. AVERAGE RATING BY CITY (Top 10 cities):")
city_ratings = df.groupby('City').agg({
    'Aggregate rating': 'mean',
    'Restaurant ID': 'count'
}).rename(columns={'Restaurant ID': 'Count'}).sort_values('Count', ascending=False).head(10)
print(city_ratings)

print("\n4. CORRELATION: Location vs Rating")
city_rating_corr = df.groupby('City')['Aggregate rating'].mean().std()
print(f"   Standard deviation of average ratings across cities: {city_rating_corr:.3f}")
print("   (Higher std dev indicates location affects rating)")

# ============================================================================
# LEVEL 2 - TASK 1: TABLE BOOKING AND ONLINE DELIVERY
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 2 - TASK 1: TABLE BOOKING AND ONLINE DELIVERY")
print("=" * 80)

# Percentage of restaurants with table booking and online delivery
table_booking_pct = (df['Has Table booking'] == 'Yes').sum() / len(df) * 100
online_delivery_pct = (df['Has Online delivery'] == 'Yes').sum() / len(df) * 100

print(f"\n1. AVAILABILITY:")
print(f"   Restaurants with Table Booking: {table_booking_pct:.2f}%")
print(f"   Restaurants with Online Delivery: {online_delivery_pct:.2f}%")

# Compare average ratings
print("\n2. AVERAGE RATINGS COMPARISON:")
booking_ratings = df.groupby('Has Table booking')['Aggregate rating'].mean()
print(f"\n   With Table Booking: {booking_ratings.get('Yes', 0):.2f}")
print(f"   Without Table Booking: {booking_ratings.get('No', 0):.2f}")

delivery_ratings = df.groupby('Has Online delivery')['Aggregate rating'].mean()
print(f"\n   With Online Delivery: {delivery_ratings.get('Yes', 0):.2f}")
print(f"   Without Online Delivery: {delivery_ratings.get('No', 0):.2f}")

# Online delivery by price range
print("\n3. ONLINE DELIVERY BY PRICE RANGE:")
delivery_by_price = df.groupby('Price range')['Has Online delivery'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
)
print(delivery_by_price)

# ============================================================================
# LEVEL 2 - TASK 2: PRICE RANGE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 2 - TASK 2: PRICE RANGE ANALYSIS")
print("=" * 80)

# Most common price range
print("\n1. PRICE RANGE DISTRIBUTION:")
price_dist = df['Price range'].value_counts().sort_index()
print(price_dist)
print(f"\n   Most Common Price Range: {price_dist.idxmax()}")

# Average rating by price range
print("\n2. AVERAGE RATING BY PRICE RANGE:")
price_ratings = df.groupby('Price range')['Aggregate rating'].mean()
print(price_ratings)

# Color with highest rating by price range
print("\n3. RATING COLOR BY PRICE RANGE:")
price_color = df.groupby(['Price range', 'Rating color']).size().unstack(fill_value=0)
print(price_color)

highest_rating_color = df.groupby('Price range').apply(
    lambda x: x['Rating color'].mode()[0] if len(x['Rating color'].mode()) > 0 else 'N/A'
)
print(f"\n   Most Common Rating Color by Price Range:")
print(highest_rating_color)

# ============================================================================
# LEVEL 2 - TASK 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 2 - TASK 3: FEATURE ENGINEERING")
print("=" * 80)

# Extract length features
df['Restaurant_Name_Length'] = df['Restaurant Name'].str.len()
df['Address_Length'] = df['Address'].str.len()

# Encode categorical variables
df['Has_Table_Booking_Encoded'] = (df['Has Table booking'] == 'Yes').astype(int)
df['Has_Online_Delivery_Encoded'] = (df['Has Online delivery'] == 'Yes').astype(int)

# Cuisine count (number of cuisine types)
df['Cuisine_Count'] = df['Cuisines'].str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 1)

print("\n1. NEW FEATURES CREATED:")
print("   - Restaurant_Name_Length")
print("   - Address_Length")
print("   - Has_Table_Booking_Encoded (0/1)")
print("   - Has_Online_Delivery_Encoded (0/1)")
print("   - Cuisine_Count")

print("\n2. FEATURE STATISTICS:")
new_features = ['Restaurant_Name_Length', 'Address_Length', 'Cuisine_Count']
print(df[new_features].describe())

# ============================================================================
# LEVEL 3 - TASK 1: PREDICTIVE MODELING
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 3 - TASK 1: PREDICTIVE MODELING")
print("=" * 80)

# Prepare features for modeling
feature_columns = [
    'Country Code', 'Average Cost for two', 'Price range',
    'Has_Table_Booking_Encoded', 'Has_Online_Delivery_Encoded',
    'Votes', 'Restaurant_Name_Length', 'Address_Length', 'Cuisine_Count'
]

# Create feature matrix and target
X = df[feature_columns].copy()
y = df['Aggregate rating'].copy()

# Handle any remaining NaN values
X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n1. DATA SPLIT:")
print(f"   Training set size: {len(X_train)}")
print(f"   Testing set size: {len(X_test)}")

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
}

results = {}

print("\n2. MODEL PERFORMANCE:")
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    print(f"\n   {name}:")
    print(f"      RMSE: {rmse:.4f}")
    print(f"      MAE: {mae:.4f}")
    print(f"      R² Score: {r2:.4f}")

# Feature importance (for Random Forest)
print("\n3. FEATURE IMPORTANCE (Random Forest):")
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# ============================================================================
# LEVEL 3 - TASK 2: CUSTOMER PREFERENCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 3 - TASK 2: CUSTOMER PREFERENCE ANALYSIS")
print("=" * 80)

# Relationship between cuisine and rating
print("\n1. AVERAGE RATING BY CUISINE TYPE:")
# Get individual cuisines
cuisine_ratings = []
for idx, row in df.iterrows():
    cuisines = str(row['Cuisines']).split(',')
    for cuisine in cuisines:
        cuisine = cuisine.strip()
        cuisine_ratings.append({
            'Cuisine': cuisine,
            'Rating': row['Aggregate rating'],
            'Votes': row['Votes']
        })

cuisine_df = pd.DataFrame(cuisine_ratings)
cuisine_stats = cuisine_df.groupby('Cuisine').agg({
    'Rating': 'mean',
    'Votes': 'sum'
}).sort_values('Votes', ascending=False).head(15)

print(cuisine_stats)

# Most popular cuisines by votes
print("\n2. MOST POPULAR CUISINES (By Total Votes):")
print(cuisine_stats['Votes'].head(10))

# Cuisines with highest ratings (minimum 50 restaurants)
print("\n3. CUISINES WITH HIGHEST RATINGS (Min 50 occurrences):")
cuisine_counts = cuisine_df['Cuisine'].value_counts()
popular_cuisines = cuisine_counts[cuisine_counts >= 50].index
high_rated = cuisine_df[cuisine_df['Cuisine'].isin(popular_cuisines)].groupby('Cuisine')['Rating'].mean().sort_values(ascending=False)
print(high_rated.head(10))

# ============================================================================
# LEVEL 3 - TASK 3: DATA VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("LEVEL 3 - TASK 3: DATA VISUALIZATION")
print("=" * 80)
print("\nGenerating visualizations...")

# Create a comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 24))

# 1. Rating Distribution
ax1 = plt.subplot(4, 3, 1)
df['Aggregate rating'].hist(bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Restaurant Ratings')
plt.axvline(df['Aggregate rating'].mean(), color='red', linestyle='--', label=f'Mean: {df["Aggregate rating"].mean():.2f}')
plt.legend()

# 2. Rating by Price Range
ax2 = plt.subplot(4, 3, 2)
df.groupby('Price range')['Aggregate rating'].mean().plot(kind='bar', color='steelblue')
plt.xlabel('Price Range')
plt.ylabel('Average Rating')
plt.title('Average Rating by Price Range')
plt.xticks(rotation=0)

# 3. Top 10 Cities by Restaurant Count
ax3 = plt.subplot(4, 3, 3)
city_counts.head(10).plot(kind='barh', color='coral')
plt.xlabel('Number of Restaurants')
plt.ylabel('City')
plt.title('Top 10 Cities by Restaurant Count')

# 4. Table Booking vs Rating
ax4 = plt.subplot(4, 3, 4)
booking_data = df.groupby('Has Table booking')['Aggregate rating'].mean()
booking_data.plot(kind='bar', color=['salmon', 'lightgreen'])
plt.xlabel('Has Table Booking')
plt.ylabel('Average Rating')
plt.title('Average Rating: Table Booking')
plt.xticks(rotation=0)

# 5. Online Delivery vs Rating
ax5 = plt.subplot(4, 3, 5)
delivery_data = df.groupby('Has Online delivery')['Aggregate rating'].mean()
delivery_data.plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.xlabel('Has Online Delivery')
plt.ylabel('Average Rating')
plt.title('Average Rating: Online Delivery')
plt.xticks(rotation=0)

# 6. Price Range Distribution
ax6 = plt.subplot(4, 3, 6)
df['Price range'].value_counts().sort_index().plot(kind='bar', color='mediumpurple')
plt.xlabel('Price Range')
plt.ylabel('Count')
plt.title('Distribution of Price Ranges')
plt.xticks(rotation=0)

# 7. Top 10 Cuisines by Count
ax7 = plt.subplot(4, 3, 7)
top_cuisines.head(10).plot(kind='barh', color='gold')
plt.xlabel('Number of Restaurants')
plt.ylabel('Cuisine')
plt.title('Top 10 Cuisines')

# 8. Rating Color Distribution
ax8 = plt.subplot(4, 3, 8)
df['Rating color'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.ylabel('')
plt.title('Rating Color Distribution')

# 9. Votes vs Rating Scatter
ax9 = plt.subplot(4, 3, 9)
plt.scatter(df['Votes'], df['Aggregate rating'], alpha=0.3, s=10)
plt.xlabel('Number of Votes')
plt.ylabel('Aggregate Rating')
plt.title('Votes vs Rating')

# 10. Average Cost vs Rating
ax10 = plt.subplot(4, 3, 10)
# Filter outliers for better visualization
cost_filtered = df[df['Average Cost for two'] < df['Average Cost for two'].quantile(0.95)]
plt.scatter(cost_filtered['Average Cost for two'], cost_filtered['Aggregate rating'], alpha=0.3, s=10, c='green')
plt.xlabel('Average Cost for Two')
plt.ylabel('Aggregate Rating')
plt.title('Cost vs Rating')

# 11. Feature Importance
ax11 = plt.subplot(4, 3, 11)
feature_importance.head(9).plot(x='Feature', y='Importance', kind='barh', ax=ax11, legend=False, color='teal')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Random Forest)')

# 12. Model Comparison
ax12 = plt.subplot(4, 3, 12)
model_names = list(results.keys())
r2_scores = [results[m]['R2'] for m in model_names]
colors_list = ['skyblue', 'lightcoral', 'lightgreen']
plt.bar(model_names, r2_scores, color=colors_list)
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.xticks(rotation=15)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('restaurant_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'restaurant_analysis_dashboard.png'")

# Additional visualizations
# Cuisine analysis
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top cuisines by average rating
axes[0, 0].barh(high_rated.head(10).index, high_rated.head(10).values, color='orchid')
axes[0, 0].set_xlabel('Average Rating')
axes[0, 0].set_ylabel('Cuisine')
axes[0, 0].set_title('Top 10 Cuisines by Rating (Min 50 restaurants)')

# Top cuisines by votes
top_vote_cuisines = cuisine_stats.head(10)
axes[0, 1].barh(top_vote_cuisines.index, top_vote_cuisines['Votes'], color='darkorange')
axes[0, 1].set_xlabel('Total Votes')
axes[0, 1].set_ylabel('Cuisine')
axes[0, 1].set_title('Top 10 Most Popular Cuisines (By Votes)')

# City average ratings
axes[1, 0].barh(city_ratings.index, city_ratings['Aggregate rating'], color='steelblue')
axes[1, 0].set_xlabel('Average Rating')
axes[1, 0].set_ylabel('City')
axes[1, 0].set_title('Average Ratings by Top Cities')

# Online delivery by price range
delivery_price = df.groupby(['Price range', 'Has Online delivery']).size().unstack()
delivery_price.plot(kind='bar', ax=axes[1, 1], stacked=True)
axes[1, 1].set_xlabel('Price Range')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Online Delivery Availability by Price Range')
axes[1, 1].legend(title='Has Online Delivery')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('cuisine_city_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Cuisine analysis saved as 'cuisine_city_analysis.png'")

# Geospatial visualization (simplified)
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

# Restaurant locations scatter plot
axes[0].scatter(df['Longitude'], df['Latitude'], 
               c=df['Aggregate rating'], cmap='RdYlGn', 
               alpha=0.6, s=20)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('Restaurant Locations (Colored by Rating)')
cbar = plt.colorbar(axes[0].collections[0], ax=axes[0])
cbar.set_label('Aggregate Rating')

# Country distribution
country_data = df['Country Code'].value_counts().head(10)
axes[1].barh(range(len(country_data)), country_data.values, color='mediumpurple')
axes[1].set_yticks(range(len(country_data)))
axes[1].set_yticklabels(country_data.index)
axes[1].set_xlabel('Number of Restaurants')
axes[1].set_ylabel('Country Code')
axes[1].set_title('Top 10 Countries by Restaurant Count')

plt.tight_layout()
plt.savefig('geospatial_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Geospatial analysis saved as 'geospatial_analysis.png'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nSummary:")
print(f"✓ Analyzed {len(df)} restaurants")
print(f"✓ Processed {len(df.columns)} features")
print(f"✓ Built and compared 3 predictive models")
print(f"✓ Generated comprehensive visualizations")
print(f"✓ Best Model: {max(results, key=lambda x: results[x]['R2'])} (R² = {max(results.values(), key=lambda x: x['R2'])['R2']:.4f})")