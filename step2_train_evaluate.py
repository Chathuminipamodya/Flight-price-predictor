import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# LOAD CLEANED DATA
# ─────────────────────────────────────────
df = pd.read_csv("flights_cleaned.csv")
print("Dataset shape:", df.shape)
print(df.head())

features = [
    "airline_enc", "destination_enc",
    "stops_num", "departure_hour",
    "day_of_week", "month", "day"
]
target = "price"

X = df[features]
y = df[target]

# ─────────────────────────────────────────
# TRAIN / VALIDATION / TEST SPLIT
# 70% train | 15% validation | 15% test
# ─────────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
)

print(f"\nTrain rows     : {len(X_train)}  ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation rows: {len(X_val)}   ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test rows      : {len(X_test)}   ({len(X_test)/len(X)*100:.1f}%)")

# ─────────────────────────────────────────
# XGBOOST MODEL
#
# WHY XGBOOST? (Write this in Section 2 of your report)
# ─ Gradient Boosting: builds trees sequentially, each correcting
#   the errors of the previous — unlike standard decision trees
# ─ Not taught in lectures (not logistic regression, kNN, etc.)
# ─ Handles non-linear feature interactions automatically
# ─ Naturally provides feature importance
# ─ Industry standard for tabular regression tasks
# ─ Regularisation (lambda, alpha) reduces overfitting
# ─────────────────────────────────────────
model = XGBRegressor(
    n_estimators=300,       # number of trees to build
    max_depth=6,            # how deep each tree can grow
    learning_rate=0.05,     # shrinks contribution of each tree (prevents overfitting)
    subsample=0.8,          # use 80% of rows per tree (adds randomness)
    colsample_bytree=0.8,   # use 80% of features per tree
    reg_alpha=0.1,          # L1 regularisation
    reg_lambda=1.0,         # L2 regularisation
    random_state=42,
    eval_metric="rmse"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# ─────────────────────────────────────────
# EVALUATE ON ALL SPLITS
# ─────────────────────────────────────────
def evaluate(name, X_split, y_split):
    preds = model.predict(X_split)
    mae  = mean_absolute_error(y_split, preds)
    rmse = np.sqrt(mean_squared_error(y_split, preds))
    r2   = r2_score(y_split, preds)
    print(f"\n📊 {name} Results:")
    print(f"   MAE  (Mean Absolute Error)  : ${mae:.2f}")
    print(f"   RMSE (Root Mean Sq. Error)  : ${rmse:.2f}")
    print(f"   R²   (Variance explained)   : {r2:.4f}")
    return preds, mae, rmse, r2

_, tr_mae, tr_rmse, tr_r2   = evaluate("TRAIN",      X_train, y_train)
_, va_mae, va_rmse, va_r2   = evaluate("VALIDATION", X_val,   y_val)
test_preds, te_mae, te_rmse, te_r2 = evaluate("TEST", X_test, y_test)

# Print comparison table
print("\n" + "="*55)
print(f"{'Split':<12} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("="*55)
print(f"{'Train':<12} {tr_mae:>10.2f} {tr_rmse:>10.2f} {tr_r2:>10.4f}")
print(f"{'Validation':<12} {va_mae:>10.2f} {va_rmse:>10.2f} {va_r2:>10.4f}")
print(f"{'Test':<12} {te_mae:>10.2f} {te_rmse:>10.2f} {te_r2:>10.4f}")
print("="*55)

# ─────────────────────────────────────────
# PLOT 1: Actual vs Predicted (Test Set)
# ─────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_preds, alpha=0.3, color="steelblue", edgecolors="none", s=20)
min_val = min(y_test.min(), test_preds.min())
max_val = max(y_test.max(), test_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.title("Actual vs Predicted Flight Prices — Test Set")
plt.legend()
plt.tight_layout()
plt.savefig("plot_actual_vs_predicted.png", dpi=150)
plt.close()
print("\n✅ Saved: plot_actual_vs_predicted.png")

# ─────────────────────────────────────────
# PLOT 2: Residuals (Errors)
# ─────────────────────────────────────────
residuals = y_test - test_preds
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=40, kde=True, color="coral")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Residual (Actual − Predicted)")
plt.title("Distribution of Prediction Errors (Test Set)")
plt.tight_layout()
plt.savefig("plot_residuals.png", dpi=150)
plt.close()
print("✅ Saved: plot_residuals.png")

# ─────────────────────────────────────────
# PLOT 3: XGBoost Feature Importance
# ─────────────────────────────────────────
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importances:")
print(feat_df.to_string(index=False))

plt.figure(figsize=(9, 5))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="Blues_d",
            hue="Feature", legend=False)
plt.title("XGBoost Feature Importance (Gain)")
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150)
plt.close()
print("✅ Saved: plot_feature_importance.png")

# ─────────────────────────────────────────
# PLOT 4: Training Curve (Validation RMSE over rounds)
# ─────────────────────────────────────────
results = model.evals_result()
rounds  = range(len(results["validation_0"]["rmse"]))
plt.figure(figsize=(8, 4))
plt.plot(rounds, results["validation_0"]["rmse"], color="darkorange", label="Validation RMSE")
plt.xlabel("Boosting Round")
plt.ylabel("RMSE")
plt.title("Validation RMSE over Boosting Rounds")
plt.legend()
plt.tight_layout()
plt.savefig("plot_training_curve.png", dpi=150)
plt.close()
print("✅ Saved: plot_training_curve.png")

# ─────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────
joblib.dump(model,    "xgboost_model.pkl")
joblib.dump(features, "feature_columns.pkl")
print("\n✅ Model saved: xgboost_model.pkl")
print("✅ Features saved: feature_columns.pkl")
