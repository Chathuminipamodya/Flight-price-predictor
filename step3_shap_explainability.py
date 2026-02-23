import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# LOAD MODEL AND DATA
# ─────────────────────────────────────────
model    = joblib.load("xgboost_model.pkl")
features = joblib.load("feature_columns.pkl")

df = pd.read_csv("flights_cleaned.csv")
X  = df[features]
y  = df["price"]

# Use 500 rows for speed (SHAP is slow on large data)
X_sample = X.sample(500, random_state=42).reset_index(drop=True)

print("Computing SHAP values... (this takes ~30 seconds)")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
print(f"✅ SHAP values computed. Shape: {shap_values.shape}")

# ─────────────────────────────────────────
# HUMAN-READABLE FEATURE NAMES (for plots)
# ─────────────────────────────────────────
feature_labels = [
    "Airline",
    "Destination",
    "No. of Stops",
    "Departure Hour",
    "Day of Week",
    "Month",
    "Day of Month"
]
X_sample_labeled = X_sample.copy()
X_sample_labeled.columns = feature_labels

shap_values_labeled = shap_values  # same values, renamed columns

# ─────────────────────────────────────────
# PLOT 1: SHAP Bar Summary
# Average impact of each feature on price prediction
# ─────────────────────────────────────────
plt.figure()
shap.summary_plot(shap_values_labeled, X_sample_labeled,
                  feature_names=feature_labels, plot_type="bar", show=False)
plt.title("SHAP Feature Importance — Mean Absolute Impact on Price")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: shap_bar.png")

# ─────────────────────────────────────────
# PLOT 2: SHAP Beeswarm (dot plot)
# Shows direction — high feature value → higher or lower price?
# ─────────────────────────────────────────
plt.figure()
shap.summary_plot(shap_values_labeled, X_sample_labeled,
                  feature_names=feature_labels, show=False)
plt.title("SHAP Beeswarm — Feature Impact Direction on Price")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: shap_beeswarm.png")

# ─────────────────────────────────────────
# PLOT 3: SHAP Waterfall for 1 example
# Explains one specific flight prediction
# ─────────────────────────────────────────
shap_expl = explainer(X_sample_labeled)
plt.figure()
shap.plots.waterfall(shap_expl[0], show=False)
plt.title("SHAP Waterfall — Why is this flight priced this way?")
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: shap_waterfall.png")

# ─────────────────────────────────────────
# PLOT 4: SHAP Dependence Plot — Stops
# How does number of stops affect price prediction?
# ─────────────────────────────────────────
plt.figure()
shap.dependence_plot("No. of Stops", shap_values_labeled, X_sample_labeled,
                     feature_names=feature_labels, show=False)
plt.title("SHAP Dependence: Number of Stops vs Price Impact")
plt.tight_layout()
plt.savefig("shap_dependence_stops.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: shap_dependence_stops.png")

# ─────────────────────────────────────────
# PRINT SHAP MEAN IMPORTANCE TABLE
# Paste this into your report Section 4
# ─────────────────────────────────────────
mean_shap = np.abs(shap_values_labeled).mean(axis=0)
shap_df = pd.DataFrame({
    "Feature": feature_labels,
    "Mean |SHAP| (USD impact)": mean_shap
}).sort_values("Mean |SHAP| (USD impact)", ascending=False)

print("\n📊 SHAP Feature Importance Table:")
print(shap_df.to_string(index=False))
print("\nCopy this table into Section 4 of your report!")

print("\n✅ All SHAP plots saved. Use them all in Section 4.")
