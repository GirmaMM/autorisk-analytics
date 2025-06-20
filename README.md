# AutoRisk Analytics: Insurance Risk Modeling Project

Welcome to the AutoRisk Analytics repository!  
This project contains the code, data, models, notebooks, and reports associated with an end-to-end insurance analytics initiative. Our objective is to explore historical claim data from the South African auto insurance market to identify risk drivers, customer segments, and opportunities for personalized premium optimization.

---

## 🚗 Project Overview

Our mission is to support AlphaCare Insurance Solutions (ACIS) in building a smarter, data-driven marketing and underwriting strategy by:

- 🔍 Identifying low-risk customer segments for targeted pricing  
- 🌍 Analyzing geographic and demographic trends in claims and profitability  
- 📊 Leveraging hypothesis testing and machine learning for risk segmentation  

The project is divided into clearly structured tasks across separate branches, each capturing a stage of the analysis pipeline — from EDA to modeling and interpretation.

---

## 📁 Project Structure (to be expanded as tasks progress)

- `data/` – DVC-tracked raw, interim, and processed datasets  
- `notebooks/` – Jupyter notebooks for EDA, hypothesis testing, and predictive modeling  
- `scripts/` – Python scripts for transformation, feature engineering, and model training  
- `tests/` – Unit tests for validating code integrity  
- `reports/` – Visualizations and final business reports  
- `.github/` – CI/CD workflows and configurations  
- `requirements.txt` – Project dependencies  
- `README.md` – This file  

---

---

## ✅ Task 1 – Exploratory Data Analysis Summary

- Processed a dataset of 1M+ insurance records with 52 columns
- Cleaned and transformed raw text data (`|`-delimited) into structured CSV
- Flagged and handled:
  - Columns with over 90% missingness
  - Negative financial values in `TotalClaims` and `TotalPremium`
  - Skewed distributions using log transformations
- Generated descriptive statistics, geographic trends, correlation matrices, and outlier detection visuals
- Used `DVC` to track raw and interim datasets, ensuring reproducibility
- Final cleaned file: `data/interim/cleaned_insurance_data.csv`

📊 EDA notebook: [`notebooks/task-1-eda.ipynb`](notebooks/task-1-eda.ipynb)

---

## ✅ Task 2 – DVC Pipeline and Versioning Summary

- 🎯 **DVC Initialized** in your project with `dvc init`  
- 🗃️ **Raw and cleaned data tracked** using `dvc add`
  - `data/row/MachineLearningRating_v3.txt`
  - `data/interim/cleaned_insurance_data.csv`
- 💾 **Local remote storage set** using `dvc remote add -d localstorage ../.dvc-storage`
- 🔁 **Pushed datasets to remote** with `dvc push`  
- 📂 **.dvc files version-controlled** with Git and committed across branches
- 📌 **Task 1 merged to main**, and a `task-2` branch used to house DVC work

---

## **✅ Task 3 – Statistical Risk Segmentation**
### **📊 Hypothesis Validation Overview**
We conducted hypothesis testing to evaluate segmentation strategies for insurance risk assessment. Key hypotheses tested:

1. **Province-Based Risk Differences**  
   - **Chi-Squared & ANOVA results:** **p < 0.001**, meaning provinces **exhibit significantly different claim behavior**.  
   - **Impact:** ACIS should adjust premium rates and underwriting policies **regionally**.

2. **Gender-Based Risk Differences**  
   - **Claim Frequency:** Chi-Squared test **p = 0.00399** → **Men and Women file claims at different rates**.  
   - **Claim Severity:** T-Test **p = 0.76** → **No significant difference in claim amounts between genders**.  
   - **Impact:** Gender-neutral pricing is advised, but claim frequency could inform risk modeling.

3. **Zip Code-Based Risk Differences** *(Feasibility still under analysis)*  
   - Unique `ZipCode` values, policy counts per zip code, and claim patterns need further validation.  
   - Next steps: Assess correlation between `ZipCode` and risk metrics.

### **🔁 Workflow & DVC Integration**
- **Git Branching:** `task-3` was used for statistical validation, merged structured commits.  
- **DVC Tracking:** All datasets are version-controlled (`.dvc` files).  
- **Reproducibility:** Hypothesis testing scripts stored in `notebooks/task-3-risk-analysis.ipynb`.  

### **📈 Business Strategy Recommendations**
✅ **Adjust premiums based on province-based risk exposure**  
✅ **Gender should not impact premium pricing, only underwriting adjustments**  
✅ **Further evaluation of zip code as a viable segmentation variable**  

📊 **Notebook & Report:** [`notebooks/task-3-risk-analysis.ipynb`](notebooks/task-3-risk-analysis.ipynb)

---
### **✅ Task 4: Predictive Modeling & Pricing Optimization**
Built **machine learning models** to predict **claims severity, premium rates, and claim probability**.
#### **Models Implemented:**
✅ **Decision Trees**  
✅ **Random Forest**  
✅ **XGBoost**  

#### **Evaluation Metrics:**
- **Claim Severity Prediction (Regression)**
  - ✅ **RMSE (Root Mean Squared Error)** → Lower RMSE = better predictions.
  - ✅ **R² Score** → Higher value = better variance explanation.
- **Claim Probability Prediction (Classification)**
  - ✅ **Accuracy, Precision, Recall, F1-score** for claim occurrence.
  
#### **Feature Importance Analysis:**
- **SHAP Explainer** → Identified **top features influencing claim predictions**.
- **Business Impact:** Dynamic pricing framework to adjust premiums based on **risk probabilities**.
- **Notebook:** [`notebooks/task-4-predictive-modeling.ipynb`](notebooks/task-4-predictive-modeling.ipynb)

---

---

## **🚀 Conclusion**
🎯 **A complete risk-based pricing solution integrating analytics, hypothesis testing, and AI modeling.**  
🧠 **Machine learning enhances risk assessment, optimizing insurance premiums dynamically.**  
📊 **Findings support actionable business decisions for ACIS.**

This repository demonstrates **end-to-end predictive risk analytics**—a scalable framework for **intelligent premium adjustments and strategic underwriting enhancements**.

---