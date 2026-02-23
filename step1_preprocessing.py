import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ─────────────────────────────────────────
# STEP 1: LOAD RAW DATA
# ─────────────────────────────────────────
df = pd.read_csv("flights_raw.csv")
print("Raw shape:", df.shape)
print(df.head(3))

# ─────────────────────────────────────────
# STEP 2: DROP ROWS WHERE KEY COLUMNS ARE MISSING
# airline, price, stops all have ~1135 NaN rows (same rows)
# ─────────────────────────────────────────
before = len(df)
df = df.dropna(subset=["airline", "price", "stops"])
print(f"\nDropped {before - len(df)} rows with missing data. Rows left: {len(df)}")

# ─────────────────────────────────────────
# STEP 3: DROP DUPLICATES
# ─────────────────────────────────────────
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows. Rows left: {len(df)}")

# ─────────────────────────────────────────
# STEP 4: CLEAN PRICE COLUMN
# Remove "$" and convert to float
# ─────────────────────────────────────────
df["price"] = df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
print(f"\nPrice range: ${df['price'].min()} – ${df['price'].max()}")

# ─────────────────────────────────────────
# STEP 5: CLEAN STOPS COLUMN
# "Nonstop" → 0, "1 stop" → 1, "2 stops" → 2, "3 stops" → 3
# ─────────────────────────────────────────
def parse_stops(s):
    s = str(s).lower().strip()
    if "nonstop" in s or "non-stop" in s:
        return 0
    elif "1" in s:
        return 1
    elif "2" in s:
        return 2
    elif "3" in s:
        return 3
    else:
        return np.nan

df["stops_num"] = df["stops"].apply(parse_stops)
df = df.dropna(subset=["stops_num"])
df["stops_num"] = df["stops_num"].astype(int)
print(f"\nStops distribution:\n{df['stops_num'].value_counts()}")

# ─────────────────────────────────────────
# STEP 6: EXTRACT DEPARTURE HOUR
# Raw format: "9:50 PM\n – \n12:50 AM+1"
# We take only the first part before \n
# ─────────────────────────────────────────
def extract_departure_hour(t):
    try:
        t = str(t).strip()
        t = t.split("\n")[0].strip()          # take first time only
        t = t.replace("\u202f", " ")          # fix narrow no-break space
        return pd.to_datetime(t, format="%I:%M %p").hour
    except:
        return np.nan

df["departure_hour"] = df["departure_time"].apply(extract_departure_hour)
df = df.dropna(subset=["departure_hour"])
df["departure_hour"] = df["departure_hour"].astype(int)
print(f"\nDeparture hour range: {df['departure_hour'].min()} – {df['departure_hour'].max()}")

# ─────────────────────────────────────────
# STEP 7: EXTRACT DATE FEATURES
# ─────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday, 6=Sunday
df["month"]       = df["date"].dt.month
df["day"]         = df["date"].dt.day
print(f"\nDate range: {df['date'].min().date()} → {df['date'].max().date()}")

# ─────────────────────────────────────────
# STEP 8: CLEAN AIRLINE NAMES
# Standardise messy names e.g. "flydubai, Emirates" → "flydubai"
# ─────────────────────────────────────────
def clean_airline(a):
    a = str(a).strip()
    if "," in a:
        a = a.split(",")[0].strip()
    return a

df["airline_clean"] = df["airline"].apply(clean_airline)
print(f"\nAirlines after cleaning:\n{df['airline_clean'].value_counts()}")

# ─────────────────────────────────────────
# STEP 9: DROP USELESS COLUMNS
# origin = always "Colombo" → no predictive value
# duration = 67% missing → dropped
# ─────────────────────────────────────────
df = df.drop(columns=["origin", "duration", "departure_time",
                       "arrival_time", "stops", "airline", "date"])

# ─────────────────────────────────────────
# STEP 10: LABEL ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────
le_airline = LabelEncoder()
le_dest    = LabelEncoder()

df["airline_enc"]     = le_airline.fit_transform(df["airline_clean"])
df["destination_enc"] = le_dest.fit_transform(df["destination"])

# Save encoder classes for Streamlit app
joblib.dump(le_airline.classes_.tolist(), "le_airline_classes.pkl")
joblib.dump(le_dest.classes_.tolist(),    "le_dest_classes.pkl")
print("\nAirline encoder classes:", le_airline.classes_.tolist())
print("Destination encoder classes:", le_dest.classes_.tolist())

# ─────────────────────────────────────────
# STEP 11: FINAL FEATURE SET
# ─────────────────────────────────────────
features = [
    "airline_enc", "destination_enc",
    "stops_num", "departure_hour",
    "day_of_week", "month", "day"
]
target = "price"

df_clean = df[features + [target]].copy()
print(f"\n✅ Final clean dataset: {df_clean.shape}")
print(df_clean.describe())

# ─────────────────────────────────────────
# STEP 12: EXPLORATORY PLOTS (use in report Section 1)
# ─────────────────────────────────────────

# Plot 1: Price distribution
plt.figure(figsize=(8, 4))
sns.histplot(df_clean["price"], bins=40, kde=True, color="steelblue")
plt.xlabel("Price (USD)")
plt.title("Flight Price Distribution")
plt.tight_layout()
plt.savefig("plot_price_distribution.png", dpi=150)
plt.close()
print("✅ Saved: plot_price_distribution.png")

# Plot 2: Price by number of stops
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="stops_num", y="price", palette="Set2")
plt.xlabel("Number of Stops")
plt.ylabel("Price (USD)")
plt.title("Flight Price by Number of Stops")
plt.tight_layout()
plt.savefig("plot_price_by_stops.png", dpi=150)
plt.close()
print("✅ Saved: plot_price_by_stops.png")

# Plot 3: Price by destination
plt.figure(figsize=(12, 5))
dest_avg = df.groupby("destination")["price"].median().sort_values(ascending=False)
sns.barplot(x=dest_avg.index, y=dest_avg.values, palette="Blues_d")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Median Price (USD)")
plt.title("Median Flight Price by Destination")
plt.tight_layout()
plt.savefig("plot_price_by_destination.png", dpi=150)
plt.close()
print("✅ Saved: plot_price_by_destination.png")

# Plot 4: Price by departure hour
plt.figure(figsize=(10, 4))
hour_avg = df.groupby("departure_hour")["price"].median().sort_index()
sns.lineplot(x=hour_avg.index, y=hour_avg.values, marker="o", color="darkorange")
plt.xlabel("Departure Hour (24h)")
plt.ylabel("Median Price (USD)")
plt.title("Median Price by Departure Hour")
plt.tight_layout()
plt.savefig("plot_price_by_hour.png", dpi=150)
plt.close()
print("✅ Saved: plot_price_by_hour.png")

# ─────────────────────────────────────────
# SAVE CLEAN DATASET
# ─────────────────────────────────────────
df_clean.to_csv("flights_cleaned.csv", index=False)
print(f"\n✅ flights_cleaned.csv saved! Final rows: {len(df_clean)}")
