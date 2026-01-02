# 📊 Financial-and-Non-Financial-Metrics-Analyzer

## 📌 Project Overview

**Financial-and-Non-Financial-Metrics-Analyzer** is an end-to-end data analytics and machine learning project designed to analyze, compare, and predict company performance using both **financial** and **non-financial** metrics.

The project integrates:
- Financial ratio analysis  
- Business performance visualization  
- Machine learning models (Regression, Classification, Clustering)  
- Automated insight generation  

## 🎯 Objectives

- Analyze and compare multiple companies over time  
- Derive key financial ratios (Profit Margin, ROA, ROE, etc.)  
- Predict **next year’s revenue** using regression  
- Classify company performance as **Good or Weak**  
- Segment company performance using **K-Means clustering**  
- Generate business insights automatically  

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy** – Data processing
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Machine learning models

---

## 📂 Dataset Description

The project expects two CSV files:

company1_data.csv
company2_data.csv


### Required Columns

| Column Name | Description |
|------------|------------|
| year | Financial year |
| revenue | Total revenue |
| cost_of_goods_sold | Cost of goods sold |
| operating_expenses | Operating expenses |
| total_assets | Total assets |
| shareholder_equity | Shareholder equity |
| customer_satisfaction | Customer satisfaction score |
| employee_turnover | Employee turnover rate |

---

## ⚙️ Feature Engineering

The following metrics are calculated:

- **Profit**  
- **Profit Margin**  
- **Operating Profit Margin**  
- **Return on Assets (ROA)**  
- **Return on Equity (ROE)**  

These engineered features are used throughout visualization, prediction, and clustering.

---

## 📈 Exploratory Data Analysis

The project generates line charts to compare companies across years for:
- Revenue  
- Profit Margin  
- ROA  
- ROE  

These plots help identify trends and performance differences.

---

## 🔮 Regression Model – Revenue Forecasting

- **Model Used:** Linear Regression  
- **Target Variable:** Next year’s revenue  
- **Evaluation Metrics:**
  - Mean Absolute Error (MAE)
  - R² Score
- **Visualization:** Actual vs Predicted revenue plot  

Purpose:  
To estimate future business growth using historical financial and operational data.

---

## 🧠 Classification Model – Performance Prediction

- **Model Used:** Random Forest Classifier  
- **Target Variable:** `good_performance`

A company-year record is labeled **Good Performance** if:
- Profit Margin > median  
- ROA > median  
- ROE > median  

### Outputs
- Classification report (Precision, Recall, F1-score)
- Confusion matrix
- Feature importance visualization

---

## 🔍 Clustering – Business Segmentation

- **Algorithm:** K-Means  
- **Number of Clusters:** 3  
- **Features Used:**
  - Profit Margin  
  - Operating Profit Margin  
  - ROA  
  - ROE  

### Output
- Cluster-wise financial summaries
- ROA vs ROE scatter plot showing company clusters

This helps identify **high-performing, moderate, and weak performance segments**.

---

## 📝 Automated Text Report

The script generates a textual summary including:
- Company with higher average profit  
- Company with higher ROA  
- Company with higher ROE  

This mimics an executive-level business summary.

---

## 💾 Output File

A processed dataset is generated:

processed_company_data_with_ratios.csv


### Contains:
- Original company data  
- Engineered financial ratios  
- Cluster labels  


