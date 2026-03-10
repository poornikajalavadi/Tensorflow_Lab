# TFX Lab 1 - Feature Engineering Pipeline

This folder contains the TFX Lab 1 notebook and modules for TensorFlow Transform-based feature engineering.

## Files
- C2_W2_Lab_2_Feature_Engineering_Pipeline.ipynb
- census_constants.py
- census_transform.py
- data/
- feature_eng_pipeline.png

## How to run
source my_tfx/bin/activate
cd TFX_Labs/TFX_Lab1
jupyter notebook

Open C2_W2_Lab_2_Feature_Engineering_Pipeline.ipynb and run cells in order.

## Dataset used

- File: data/AI Job Market Dataset.csv
- Format: CSV
- Total lines: 10,346 (10,345 data rows + 1 header row)
- Total columns: 19

### Dataset columns

job_id, job_title, company_size, company_industry, country, remote_type, experience_level, years_experience, education_level, skills_python, skills_sql, skills_ml, skills_deep_learning, skills_cloud, salary, job_posting_month, job_posting_year, hiring_urgency, job_openings

### How this lab uses the dataset

- Prediction target (label): salary
- Categorical features: job_title, company_size, company_industry, country, remote_type, experience_level, education_level
- Numeric features: skills_python, skills_sql, skills_ml, skills_deep_learning, skills_cloud, job_posting_month, job_posting_year, job_openings
- Bucketized feature: years_experience
