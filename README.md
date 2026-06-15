# Restaurant Analysis 🍽️

End-to-end data analysis project completed as part of the Cognifyz Technologies Data Science Internship. The project explores a global restaurant dataset (~9,500 records) to uncover insights about ratings, cuisines, locations, and customer preferences, and builds predictive models for restaurant ratings.

## 📊 Dataset

The dataset contains restaurant-level details including location (city, locality, latitude/longitude), cuisines, price range, average cost for two, table booking and online delivery availability, and aggregate ratings/votes.

## 🔍 Project Tasks

### Level 1
- **Data Exploration & Preprocessing** — handled missing values, checked data types, and explored target variable distribution
- **Descriptive Analysis** — statistical summary of numerical columns, distribution of categories (cities, cuisines)
- **Geospatial Analysis** — visualized restaurant locations and explored the relationship between location and ratings

- ### Level 2
- - **Table Booking & Online Delivery Analysis** — compared ratings and pricing across restaurants offering these services
- **Price Range Analysis** — examined how price range relates to ratings and ratings distribution
- **Feature Engineering** — created new features from existing columns to improve model performance

### Level 3
- **Predictive Modeling** — built and evaluated multiple regression models (Linear Regression, Decision Tree, Random Forest) to predict restaurant ratings, using R² and MAE as evaluation metrics
- **Customer Preference Analysis** — analyzed how cuisines, price range, and services impact customer ratings
- **Data Visualization** — created plots and charts to communicate findings clearly

## 🛠️ Tools & Libraries

Python, Pandas, NumPy, Matplotlib/Seaborn, Scikit-learn

## 🚀 How to Run

~~~bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py
~~~

## 📌 Key Takeaways

- Identified key factors influencing restaurant ratings
- Built regression models capable of predicting ratings based on restaurant attributes
- Delivered geospatial and price-based insights useful for business decision-making



