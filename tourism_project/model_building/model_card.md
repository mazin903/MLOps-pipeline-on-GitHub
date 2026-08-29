# Tourism Package Prediction Model

This model predicts whether a customer is likely to purchase the Wellness Tourism Package.

## Model

- Algorithm: XGBoost binary classifier
- Preprocessing: numeric median imputation, standard scaling, categorical mode imputation, one-hot encoding
- Target: `ProdTaken`
- Decision use: prioritize sales outreach and campaign targeting

## Training

Training uses stratified train/test splitting, class imbalance weighting, grid-search hyperparameter tuning, threshold analysis, and MLflow experiment tracking.

## Evaluation

Local validation generated these representative test metrics:

- Accuracy: 0.913
- Precision: 0.764
- Recall: 0.794
- F1 score: 0.778

## Limitations

The model should support human sales decisions rather than fully automate customer treatment. Predictions should be monitored over time because campaign strategy, package positioning, and customer behavior can change.

