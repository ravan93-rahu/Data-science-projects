# Descriptive Analytics and Data Preprocessing on Sales & Discounts Dataset

This project demonstrates basic statistical analysis, data visualization, and preprocessing techniques on a Sales & Discounts dataset using Python and pandas.

##  Objectives

- Perform descriptive analytics on numerical and categorical data
- Visualize data distributions and relationships
- Standardize numerical variables
- Convert categorical variables into dummy variables for machine learning

---

## Descriptive Analytics

Basic statistical measures were calculated for numerical columns:

- **Mean**: Average value
- **Median**: Middle value
- **Mode**: Most frequent value
- **Standard Deviation**: Spread of the data

These metrics help understand the central tendency and variability of the dataset.

---

## Data Visualization

### Histograms
Visualized the distribution of numerical columns to detect skewness and outliers.

### Boxplots
Used to identify outliers and understand the interquartile range (IQR).

### Bar Charts
Created for categorical columns to analyze the frequency of each category.

---

## Standardization

Applied **z-score normalization** to scale numerical variables:



\[
z = \frac{x - \mu}{\sigma}
\]



This ensures all features contribute equally to machine learning models.

---

##  One-Hot Encoding

Converted categorical columns into binary (0/1) columns using **one-hot encoding** to make them suitable for ML algorithms.

---

##  Conclusion

- Descriptive analytics provided insights into data distribution and variability.
- Visualizations helped detect outliers and category imbalances.
- Standardization and one-hot encoding prepared the dataset for modeling.

---

## 🛠 Tools Used

- Python
- pandas
- matplotlib / seaborn
- scikit-learn

---


