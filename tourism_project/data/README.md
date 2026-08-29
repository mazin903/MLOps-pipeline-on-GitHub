---
pretty_name: Tourism Package Prediction Dataset
language:
  - en
tags:
  - tabular-classification
  - tourism
  - mlops
  - sales-analytics
task_categories:
  - tabular-classification
size_categories:
  - 1K<n<10K
---

# Tourism Package Prediction Dataset

This dataset contains customer profile and sales interaction records for Visit with Us. The target column is `ProdTaken`, where `1` means the customer purchased the offered package and `0` means the customer did not.

## Files

- `tourism.csv`: raw source dataset
- `processed/cleaned_tourism.csv`: cleaned modeling dataset
- `processed/Xtrain.csv`, `processed/Xtest.csv`: feature splits
- `processed/ytrain.csv`, `processed/ytest.csv`: target splits

## Intended Use

The dataset supports supervised classification for sales prioritization. It should be used to help marketing and sales teams identify customers who are more likely to purchase the Wellness Tourism Package before outreach.

## Cleaning Summary

The pipeline removes generated index and customer identifier columns, standardizes categorical values, imputes missing numeric values with medians, imputes missing categorical values with the mode, removes duplicates, and creates stratified train/test splits.
