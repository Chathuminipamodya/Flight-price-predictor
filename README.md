# ✈️ Flight Price Predictor — MSc AI Assignment

## Files

| File | Purpose | Assignment Section |
|---|---|---|
| `flights_raw.csv` | Your scraped dataset | — |
| `step1_preprocessing.py` | Clean data, EDA plots | Section 1 (15 marks) |
| `step2_train_evaluate.py` | Train XGBoost, evaluate | Sections 2+3 (35 marks) |
| `step3_shap_explainability.py` | SHAP explanation plots | Section 4 (20 marks) |
| `app.py` | Streamlit web app | Bonus (10 marks) |

---

## How to Run

### 1. Install all libraries
```bash
pip install -r requirements.txt
```

### 2. Run in this exact order

```bash
# Step 1: Clean the data
python step1_preprocessing.py
# → Creates: flights_cleaned.csv + 4 EDA plots

# Step 2: Train the model
python step2_train_evaluate.py
# → Creates: xgboost_model.pkl + 4 evaluation plots

# Step 3: SHAP explainability
python step3_shap_explainability.py
# → Creates: 4 SHAP plots

# Step 4: Launch the web app (bonus)
streamlit run app.py
# → Opens in your browser at http://localhost:8501
```

---

## Dataset Summary

- **Source**: Scraped from Google Flights (Colombo departures)
- **Raw rows**: 3,939
- **After cleaning**: 2,687
- **Routes**: Colombo → 18 destinations
- **Airlines**: 31 airlines
- **Target variable**: Price (USD)
- **Features used**: Airline, Destination, Stops, Departure Hour, Day of Week, Month, Day

---

## Algorithm: XGBoost (eXtreme Gradient Boosting)

**Why XGBoost?**
- Not taught in lectures (not decision tree, kNN, logistic regression, etc.)
- Gradient Boosting: trains trees sequentially, each correcting the previous tree's errors
- Handles non-linear feature interactions automatically
- Industry-standard for tabular regression problems
- Has built-in regularisation to prevent overfitting
- Naturally supports SHAP for explainability

---

## Plots Generated

### Section 1 — EDA
- `plot_price_distribution.png` — Price histogram
- `plot_price_by_stops.png` — Price vs stops boxplot
- `plot_price_by_destination.png` — Median price per destination
- `plot_price_by_hour.png` — Price trends by hour

### Section 3 — Model Evaluation
- `plot_actual_vs_predicted.png` — Test set predictions
- `plot_residuals.png` — Error distribution
- `plot_feature_importance.png` — XGBoost feature importances
- `plot_training_curve.png` — RMSE over boosting rounds

### Section 4 — Explainability (SHAP)
- `shap_bar.png` — Global feature importance
- `shap_beeswarm.png` — Impact direction per feature
- `shap_waterfall.png` — Single prediction explained
- `shap_dependence_stops.png` — Stops vs price relationship
