# ipl-win-predictor

Predicting IPL match winners using machine learning on ball-by-ball data (2008–2024).

## Tech Stack
- Python, Pandas, Scikit-learn, XGBoost, Streamlit

## Dataset
- IPL Complete Dataset from Kaggle (matches.csv + deliveries.csv)

## Progress
- [x] Day 1 — EDA on matches.csv (win counts, toss analysis, season trends)

- [x] Day 2 — Loaded deliveries.csv, explored ball-by-ball data, fixed folder structure

- [x] Day 3 — EDA visualizations, matplotlib & seaborn(02_visualizations.ipynb)

- [x] Day 4 — ML model with Logistic Regression, train/test split, LabelEncoder (03_logistic_regression.ipynb)

- [x] Day 6 — Feature engineering: One Hot Encoding, Label Encoding, X/y split, train/test split (04_feature_engineering.ipynb)

- [x] Day 7 — Trained Random Forest model, 50% accuracy on pre-match features (matches.csv)

- [x] Day 8 - In-match feature engineering from deliveries.csv (run rate, wickets remaining, required run rate), merged with matches.csv, Random Forest accuracy: 88.8% (05_inplay_features.ipynb)

- [x] Day 9 — XGBoost vs Random Forest comparison (RF: 88.8%, XGBoost: 85.3%), feature importance analysis

- [x] Day 10 — Feature importance, overfitting fix (train: 85.7%, test: 81%)

- [x] Day 11-15 — In-match feature engineering, data leakage fix (GroupShuffleSplit), overfitting fix (train: 85.7%, test: 81%), confusion matrix, saved model as pkl

- [x] Phase 3 (in progress) — Streamlit app built, loads pkl model, predicts live win probability