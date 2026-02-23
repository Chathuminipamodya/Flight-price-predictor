

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG — must be first
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0710;
    color: #e8eaf0;
}
.stApp { background: #0a0710; }

.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1a3a 50%, #071428 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "✈";
    position: absolute;
    right: 40px; top: 50%;
    transform: translateY(-50%);
    font-size: 120px;
    opacity: 0.04;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    color: #ffffff; margin: 0 0 8px 0; letter-spacing: -1px;
}
.hero p { color: #8899bb; font-size: 1.05rem; margin: 0; font-weight: 300; }
.hero .badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #b794f4; padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 500; margin-bottom: 16px; letter-spacing: 0.5px;
}

.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #b794f4; letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 16px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}

.card {
    background: #12092b; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 24px; margin-bottom: 16px;
}

.price-result {
    background: linear-gradient(135deg, #0a2545 0%, #0d3060 100%);
    border: 1px solid rgba(99,179,237,0.4); border-radius: 20px;
    padding: 32px 40px; text-align: center; margin: 24px 0;
    box-shadow: 0 0 40px rgba(99,179,237,0.08);
}
.price-result .label {
    font-size: 0.85rem; color: #8899bb; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 500; margin-bottom: 8px;
}
.price-result .amount {
    font-family: 'Syne', sans-serif; font-size: 4rem; font-weight: 800;
    color: #b794f4; line-height: 1; margin-bottom: 8px;
}
.price-result .sub { color: #8899bb; font-size: 0.9rem; }

.stat-box {
    background: #12092b; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 20px; text-align: center;
}
.stat-box .stat-val {
    font-family: 'Syne', sans-serif; font-size: 1.6rem;
    font-weight: 700; color: #b794f4;
}
.stat-box .stat-label {
    font-size: 0.78rem; color: #8899bb;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;
}

.info-row {
    display: flex; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.92rem;
}
.info-row .key { color: #8899bb; }
.info-row .val { color: #e8eaf0; font-weight: 500; }

.gauge-wrap { margin: 8px 0; }
.gauge-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: #8899bb; margin-bottom: 4px; }
.gauge-bar { height: 8px; background: #1a2035; border-radius: 4px; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #2b6cb0, #b794f4); }

.stSelectbox label, .stSlider label, .stDateInput label {
    color: #8899bb !important; font-size: 0.82rem !important;
    font-weight: 500 !important; letter-spacing: 0.5px !important; text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background-color: #0d1627 !important;
    border-color: rgba(99,179,237,0.2) !important;
    border-radius: 10px !important; color: #e8eaf0 !important;
}
.stButton > button {
    background: linear-gradient(135deg,  #553c9a, #2b7fd4) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 14px 32px !important; font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    width: 100% !important; box-shadow: 0 4px 20px rgba(43,127,212,0.3) !important;
}
.stTabs [data-baseweb="tab"] { color: #8899bb !important; }
.stTabs [aria-selected="true"] { color: #b794f4 !important; border-bottom-color: #b794f4 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_all():
    model    = joblib.load("xgboost_model.pkl")
    features = joblib.load("feature_columns.pkl")
    airlines = joblib.load("le_airline_classes.pkl")
    dests    = joblib.load("le_dest_classes.pkl")
    df_clean = pd.read_csv("flights_cleaned.csv")
    df_raw   = pd.read_csv("flights_raw.csv")
    df_raw   = df_raw.dropna(subset=["airline","price","stops"])
    df_raw["price_usd"] = pd.to_numeric(
        df_raw["price"].str.replace(r"[^\d.]","",regex=True), errors="coerce")
    df_raw = df_raw.dropna(subset=["price_usd"])
    return model, features, airlines, dests, df_clean, df_raw

model, features, AIRLINES, DESTINATIONS, df_clean, df_raw = load_all()

def encode(value, options):
    options = sorted(options)
    return options.index(value) if value in options else 0

# matplotlib dark theme
plt.rcParams.update({
    "figure.facecolor":"#12092b","axes.facecolor":"#12092b",
    "axes.edgecolor":"#1e2d45","axes.labelcolor":"#8899bb",
    "axes.titlecolor":"#e8eaf0","xtick.color":"#8899bb",
    "ytick.color":"#8899bb","text.color":"#e8eaf0",
    "grid.color":"#1a2540","grid.alpha":0.5,
})

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">✈ MSc AI · Machine Learning Assignment</div>
    <h1>Flight Price Predictor</h1>
    <p>XGBoost model trained on 2,687 real scraped flights from Colombo · Explained with SHAP</p>
</div>""", unsafe_allow_html=True)

# ─── STAT STRIP ───
c1,c2,c3,c4,c5 = st.columns(5)
for col, val, label in zip(
    [c1,c2,c3,c4,c5],
    ["2,687","18","31",f"${int(df_raw['price_usd'].min())}",f"${int(df_raw['price_usd'].max()):,}"],
    ["Flights Trained","Destinations","Airlines","Lowest Price","Highest Price"]
):
    with col:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown('<div class="section-title">Flight Details</div>', unsafe_allow_html=True)
    airline        = st.selectbox("Airline", sorted(AIRLINES))
    destination    = st.selectbox("Destination", sorted(DESTINATIONS))
    stops          = st.selectbox("Number of Stops", [0,1,2,3],
                                  format_func=lambda x:["Nonstop","1 Stop","2 Stops","3 Stops"][x])
    flight_date    = st.date_input("Flight Date")
    departure_hour = st.slider("Departure Hour", 0, 23, 9, format="%d:00")

    time_label = ("🌙 Red-eye" if departure_hour < 6 else
                  "🌅 Morning" if departure_hour < 12 else
                  "☀️ Afternoon" if departure_hour < 17 else
                  "🌆 Evening" if departure_hour < 21 else "🌙 Night")
    st.caption(f"Departure: **{departure_hour:02d}:00** — {time_label}")
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Flight Price", use_container_width=True)

with right:
    if predict_btn:
        input_dict = {
            "airline_enc"    : encode(airline, AIRLINES),
            "destination_enc": encode(destination, DESTINATIONS),
            "stops_num"      : stops,
            "departure_hour" : departure_hour,
            "day_of_week"    : flight_date.weekday(),
            "month"          : flight_date.month,
            "day"            : flight_date.day,
        }
        input_df = pd.DataFrame([input_dict])[features]
        predicted_price = float(model.predict(input_df)[0])
        day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        st.markdown(f"""
        <div class="price-result">
            <div class="label">Estimated Ticket Price</div>
            <div class="amount">${predicted_price:,.0f}</div>
            <div class="sub">Colombo → {destination} · {airline}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Flight Summary</div>', unsafe_allow_html=True)
        for k, v in [
            ("Route",      f"Colombo → {destination}"),
            ("Airline",    airline),
            ("Date",       f"{flight_date.strftime('%d %B %Y')} ({day_names[flight_date.weekday()]})"),
            ("Departure",  f"{departure_hour:02d}:00 — {time_label}"),
            ("Stops",      ["Nonstop","1 Stop","2 Stops","3 Stops"][stops]),
        ]:
            st.markdown(f'<div class="info-row"><span class="key">{k}</span><span class="val">{v}</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # price gauge
        dest_prices = df_raw[df_raw["destination"]==destination]["price_usd"]
        if len(dest_prices) > 0:
            mn, med, mx = dest_prices.min(), dest_prices.median(), dest_prices.max()
            pct = float(np.clip((predicted_price-mn)/(mx-mn)*100,0,100))
            st.markdown('<div class="section-title">Price Position</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="gauge-wrap">
                <div class="gauge-label">
                    <span>Min ${mn:,.0f}</span>
                    <span>Your price ${predicted_price:,.0f}</span>
                    <span>Max ${mx:,.0f}</span>
                </div>
                <div class="gauge-bar"><div class="gauge-fill" style="width:{pct:.1f}%"></div></div>
            </div>""", unsafe_allow_html=True)

            sc1,sc2,sc3 = st.columns(3)
            for col,v,lbl,clr in zip(
                [sc1,sc2,sc3],
                [f"${mn:,.0f}", f"${med:,.0f}", f"${mx:,.0f}"],
                ["Min Price","Median Price","Max Price"],
                ["#48bb78","#b794f4","#fc8181"]
            ):
                with col:
                    st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:{clr}">{v}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px 24px;">
            <div style="font-size:3rem;margin-bottom:16px;">✈️</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#e8eaf0;margin-bottom:8px;">Ready to Predict</div>
            <div style="color:#8899bb;font-size:0.9rem;">Fill in flight details on the left<br>and click Predict to see the estimated price</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Analysis & Insights</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Analysis", "🔍 SHAP Explanation", "📈 Model Performance", "🗺️ Dataset Explorer"
])

# ── TAB 1 ──
with tab1:
    g1,g2 = st.columns(2)
    with g1:
        fig,ax = plt.subplots(figsize=(7,5))
        dest_avg = df_raw.groupby("destination")["price_usd"].median().sort_values()
        clrs = ["#f6ad55" if d==destination else "#2b6cb0" for d in dest_avg.index]
        bars = ax.barh(dest_avg.index, dest_avg.values, color=clrs, height=0.6)
        ax.set_xlabel("Median Price (USD)")
        ax.set_title("Median Price by Destination", fontweight="bold", pad=15)
        ax.grid(axis="x",alpha=0.3)
        ax.spines[["top","right","left"]].set_visible(False)
        for bar,val in zip(bars,dest_avg.values):
            ax.text(val+10, bar.get_y()+bar.get_height()/2, f"${val:,.0f}",
                    va="center", fontsize=7.5, color="#8899bb")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with g2:
        fig,ax = plt.subplots(figsize=(7,5))
        stop_labels = ["Nonstop","1 stop","2 stops","3 stops"]
        stop_data = [df_raw[df_raw["stops"]==s]["price_usd"].dropna().values for s in stop_labels]
        bp = ax.boxplot(stop_data, patch_artist=True,
                        medianprops=dict(color="#f6ad55",linewidth=2),
                        whiskerprops=dict(color="#8899bb"),capprops=dict(color="#8899bb"),
                        flierprops=dict(marker="o",markerfacecolor="#b794f4",markersize=3,alpha=0.4))
        for patch,c in zip(bp["boxes"],["#1a3a5c","#1e4976","#234f7a","#27567e"]):
            patch.set_facecolor(c); patch.set_edgecolor("#b794f4")
        ax.set_xticklabels(["Nonstop","1 Stop","2 Stops","3 Stops"])
        ax.set_ylabel("Price (USD)")
        ax.set_title("Price by Number of Stops", fontweight="bold", pad=15)
        ax.grid(axis="y",alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    g3,g4 = st.columns(2)
    with g3:
        fig,ax = plt.subplots(figsize=(7,4))
        ax.hist(df_raw["price_usd"], bins=45, color="#2b6cb0",
                edgecolor="#12092b", alpha=0.85, linewidth=0.5)
        if predict_btn:
            ax.axvline(predicted_price, color="#f6ad55", linewidth=2.5,
                       label=f"Your prediction: ${predicted_price:,.0f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("Price (USD)"); ax.set_ylabel("Number of Flights")
        ax.set_title("Overall Price Distribution", fontweight="bold", pad=15)
        ax.grid(axis="y",alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with g4:
        fig,ax = plt.subplots(figsize=(7,4))
        airline_med = df_raw.groupby("airline")["price_usd"].median().sort_values(ascending=False).head(10)
        colors = ["#f6ad55" if a==airline else "#2b6cb0" for a in airline_med.index]
        ax.barh(airline_med.index, airline_med.values, color=colors, height=0.6)
        ax.set_xlabel("Median Price (USD)")
        ax.set_title("Top 10 Airlines by Median Price", fontweight="bold", pad=15)
        ax.grid(axis="x",alpha=0.3)
        ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 2 ──
with tab2:
    if predict_btn:
        st.markdown("#### Why was **$" + f"{predicted_price:,.0f}** predicted for this flight?")
        st.markdown("SHAP shows which features pushed the price **up** 🔴 or **down** 🔵.")
        feature_labels = ["Airline","Destination","No. of Stops","Departure Hour","Day of Week","Month","Day of Month"]
        input_labeled = input_df.copy(); input_labeled.columns = feature_labels
        explainer = shap.TreeExplainer(model)
        shap_expl = explainer(input_labeled)
        sample = df_clean[features].sample(min(300,len(df_clean)),random_state=42)
        sv = explainer.shap_values(sample)
        sample_labeled = sample.copy(); sample_labeled.columns = feature_labels

        sh1,sh2 = st.columns(2)
        with sh1:
            fig,_ = plt.subplots(figsize=(7,5))
            shap.plots.waterfall(shap_expl[0],show=False)
            plt.title("This Prediction Explained",fontsize=11,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with sh2:
            fig,_ = plt.subplots(figsize=(7,5))
            shap.summary_plot(sv,sample_labeled,feature_names=feature_labels,plot_type="bar",show=False)
            plt.title("Global Feature Importance",fontsize=11,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        fig,_ = plt.subplots(figsize=(12,4))
        shap.summary_plot(sv,sample_labeled,feature_names=feature_labels,show=False)
        plt.title("SHAP Beeswarm — Impact Direction",fontsize=11,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        mean_shap = np.abs(sv).mean(axis=0)
        st.dataframe(pd.DataFrame({"Feature":feature_labels,"Mean SHAP Impact (USD)":mean_shap.round(2)})
                     .sort_values("Mean SHAP Impact (USD)",ascending=False).reset_index(drop=True),
                     use_container_width=True, hide_index=True)
    else:
        st.info("👈 Click **Predict** first to see the SHAP explanation for your specific flight.")

# ── TAB 3 ──
with tab3:
    st.markdown("#### Model Evaluation — XGBoost Regressor")
    m1,m2,m3,m4 = st.columns(4)
    for col,v,lbl,desc in zip([m1,m2,m3,m4],
        ["~0.82","~$98","~$180","403"],
        ["R² Score","MAE","RMSE","Test Rows"],
        ["Variance explained","Avg prediction error","Root mean sq. error","Unseen test samples"]):
        with col:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{v}</div><div class="stat-label">{lbl}</div><div style="color:#8899bb;font-size:0.72rem;margin-top:6px">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    try:
        p1,p2 = st.columns(2)
        with p1: st.image("plot_actual_vs_predicted.png", caption="Actual vs Predicted", use_container_width=True)
        with p2: st.image("plot_feature_importance.png", caption="Feature Importance", use_container_width=True)
        p3,p4 = st.columns(2)
        with p3: st.image("plot_residuals.png", caption="Residual Distribution", use_container_width=True)
        with p4: st.image("plot_training_curve.png", caption="Training Curve", use_container_width=True)
    except:
        st.info("Run `python step2_train_evaluate.py` first to generate evaluation plots.")

# ── TAB 4 ──
with tab4:
    st.markdown("#### Explore Your Scraped Dataset")
    e1,e2 = st.columns(2)
    with e1: sel_dest = st.selectbox("Filter by Destination", ["All"]+sorted(df_raw["destination"].dropna().unique()))
    with e2: sel_stop = st.selectbox("Filter by Stops", ["All","Nonstop","1 stop","2 stops","3 stops"])

    filtered = df_raw.copy()
    if sel_dest != "All": filtered = filtered[filtered["destination"]==sel_dest]
    if sel_stop != "All": filtered = filtered[filtered["stops"]==sel_stop]

    st.markdown(f"**{len(filtered):,} flights** match your filter")
    st.dataframe(filtered[["airline","destination","stops","price"]].dropna().reset_index(drop=True).head(100),
                 use_container_width=True, hide_index=True)

    if sel_dest != "All" and len(filtered) > 0:
        fig,ax = plt.subplots(figsize=(10,4))
        air_avg = filtered.groupby("airline")["price_usd"].median().sort_values(ascending=False).head(12)
        ax.bar(range(len(air_avg)), air_avg.values, color="#2b6cb0", width=0.6)
        ax.set_xticks(range(len(air_avg)))
        ax.set_xticklabels(air_avg.index, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Median Price (USD)")
        ax.set_title(f"Airline Prices to {sel_dest}", fontweight="bold")
        ax.grid(axis="y",alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# FOOTER
st.markdown("""
<div style="text-align:center;padding:32px 0 16px;color:#3a4a6b;font-size:0.8rem;">
    MSc in Artificial Intelligence · Machine Learning Assignment<br>
    XGBoost Regressor · SHAP Explainability · Trained on 2,687 real scraped flights from Colombo
</div>""", unsafe_allow_html=True)
