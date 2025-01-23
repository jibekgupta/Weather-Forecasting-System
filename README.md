# Forecasting Weather Patterns with Ridge Regression

## Project Overview
Accurately predicting weather patterns is essential for agriculture, disaster management, and everyday planning. This project utilizes Ridge Regression, a linear regression model with regularization, to forecast weather conditions based on historical data. The approach focuses on data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation to deliver accurate predictions.

---

## Table of Contents
1. [Technologies Used](#technologies-used)
2. [Data Overview](#data-overview)
3. [Project Workflow](#project-workflow)
    - [1. Importing Libraries](#1-importing-libraries)
    - [2. Data Inspection](#2-data-inspection)
    - [3. Missing Value Handling](#3-missing-value-handling)
    - [4. Feature Selection and Renaming](#4-feature-selection-and-renaming)
    - [5. Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
    - [6. Ridge Regression Modeling](#6-ridge-regression-modeling)
    - [7. Feature Engineering](#7-feature-engineering)
4. [Results and Insights](#results-and-insights)
5. [Future Enhancements](#future-enhancements)
6. [How to Run](#how-to-run)
7. [License](#license)

---

## Technologies Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn
- **Machine Learning Model**: Ridge Regression
- **Environment**: Jupyter Notebook, Visual Studio Code

---

## Data Overview
The dataset consists of historical weather data from the **Washington Reagan National Airport** (DCA), containing 49 columns of meteorological measurements spanning over seven decades. Key variables include:
- `precip`: Precipitation
- `snow`: Snowfall
- `snow_depth`: Snow depth
- `temp_max`: Maximum temperature
- `temp_min`: Minimum temperature

---

## Project Workflow

### 1. Importing Libraries
Essential Python libraries were imported to facilitate data manipulation, visualization, and machine learning:
```python
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
```
***
### 2. Data Inspection
The dataset was loaded and inspected for missing values, data types, and general structure:
```python
weather = pd.read_csv('DCA_airport.csv', index_col='DATE')
weather.info()
weather.head()
```
***
### 3. Missing Value Handling
Missing data were addressed using strategies like forward-filling for snow_depth and imputing with the mean for other variables:
```python
weather_core['snow_depth'] = weather_core['snow_depth'].ffill()
imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train[predictors])
```
***
### 4. Feature Selection and Renaming
Key features were selected for analysis:
- Relevant columns: `precip`, `snow`, `snow_depth`, `temp_max`, `temp_min`
- Columns renamed for readability:
  - `precip` → Precipitation
  - `temp_max` → Max Temperature
  - `temp_min` → Min Temperature
***
### 5. Exploratory Data Analysis (EDA)
Key patterns and trends were visualized:
- **Temperature Trends:** Observed seasonal variations in maximum and minimum temperatures.
- **Precipitation Patterns:** Summarized annual precipitation levels.
```python
plt.figure(figsize=(12,6))
weather_core[['temp_max', 'temp_min']].plot()
plt.title('Temperature Trends')
plt.show()
```
***
### 6. Ridge Regression Modeling
The Ridge Regression model was trained to predict the next day's maximum temperature based on historical weather features. **Mean Absolute Error (MAE)** was calculated to evaluate performance:
- Baseline MAE: `5.46`
```python
reg = Ridge(alpha=0.1)
reg.fit(train_imputed, train['target'])
predictions = reg.predict(test_imputed)
error = mean_absolute_error(test['target'], predictions)
print(f"Mean Absolute Error: {error}")
```
***
### 7. Feature Engineering
Additional predictors were engineered to improve the model:
- Rolling averages (e.g., monthly max temperature)
- Ratios between maximum and minimum temperatures
Resulting MAE: `5.33`
```python
weather_core['month_max'] = weather_core['temp_max'].rolling(30).mean()
error, combined = create_predictions(predictors, weather_core, reg)
print(f"Mean Absolute Error with additional predictors: {error}")
```

---

## Results and Insights
- **Performance:** Ridge Regression effectively modeled temperature predictions, achieving an MAE of 5.33 after feature engineering.
- **Feature Impact:** Engineered features, such as rolling averages, enhanced model performance.
- **Visualization:** Plots of actual vs. predicted values showed a close fit, demonstrating the model's effectiveness.
  
---

## Future Enhancements
1. Incorporate additional features like humidity or wind speed for improved accuracy.
2. Experiment with other regression models (e.g., Lasso, ElasticNet).
3. Explore hyperparameter optimization techniques like Grid Search or Random Search.
4. Expand the dataset to include data from multiple weather stations for broader generalization.
***
## How to Run
1. Clone the repository:
2. Navigate to the project directory:
3. Install dependencies: 
4. Run the Jupyter Notebook or Python script to train the model and generate predictions.

***

##  License
This project is licensed under the MIT License. See the LICENSE file for more details.
**Everything looks good now! Let me know if you'd like to make any more changes. 😊**



   



