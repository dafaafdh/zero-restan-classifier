"""
C_dashboard.py — Streamlit Dashboard
PTPN IV PalmCo — Sistem Monitoring & Prediksi Restan Buah Sawit

Jalankan dengan:
    streamlit run C_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dashboard Restan PTPN IV",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

HIJAU = "#2E7D32"
MERAH = "#C62828"
BIRU  = "#1565C0"
HARI_LABEL = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
BULAN_LABEL = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
               7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
FEATURES = [
    "Jumlah Pemanen","% Langsir","Curah Hujan",
    "Jumlah Trip/Hari","Rata-rata Ritase/Hari",
    "Jam Timbang Pertama","Bulan","DayOfWeek",
]

# ── CSS kustom ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #666; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; }
    .pred-box {
        border-radius: 14px;
        padding: 22px 28px;
        text-align: center;
        margin-top: 14px;
    }
    .pred-restan  { background: #FFEBEE; border: 2px solid #C62828; }
    .pred-aman    { background: #E8F5E9; border: 2px solid #2E7D32; }
    .section-title { font-size: 18px; font-weight: 700; margin: 8px 0 4px; }
    .insight-box {
        background: #FFF8E1;
        border-left: 5px solid #F9A825;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("data_clean.csv", parse_dates=["Waktu"])
    return df

@st.cache_resource
def load_models():
    rf = joblib.load("model_rf.pkl")
    dt = joblib.load("model_dt.pkl")
    return rf, dt

df = load_data()
rf_model, dt_model = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/PTPN_IV_logo.svg/240px-PTPN_IV_logo.svg.png",
             width=140)
    st.markdown("## 🌴 Dashboard Restan")
    st.markdown("**PTPN IV PalmCo**")
    st.markdown("Sistem Monitoring & Prediksi Restan Buah Sawit")
    st.divider()
    halaman = st.radio(
        "Navigasi",
        ["📊 Overview", "🔍 Eksplorasi Data", "🤖 Prediksi Restan", "💡 Rekomendasi"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: Jan – Jun 2025 | n = 175 hari kerja")
    st.caption("Model: Random Forest + Decision Tree")

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if halaman == "📊 Overview":
    st.title("📊 Overview — Monitoring Restan Harian")
    st.caption("Gambaran umum kondisi restan buah sawit selama periode Januari – Juni 2025")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total_restan   = df["Restan"].sum()
    rata_restan    = df[df["Restan"] > 0]["Restan"].mean()
    pct_hari_restan= (df["Restan_Binary"].mean()) * 100
    hari_terberat  = df.loc[df["Restan"].idxmax(), "Waktu"].strftime("%d %b %Y")

    for col, label, value, color in zip(
        [col1, col2, col3, col4],
        ["Total Restan (kg)", "Rata-rata/Hari Restan (kg)",
         "% Hari Ada Restan", "Hari Restan Terberat"],
        [f"{total_restan:,.0f}", f"{rata_restan:,.0f}",
         f"{pct_hari_restan:.1f}%", hari_terberat],
        [MERAH, MERAH, MERAH, BIRU],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tren harian ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Tren Restan & Produksi Harian</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
    fig.patch.set_facecolor("#FAFAFA")
    for ax, col, color, label in zip(
        axes,
        ["Restan", "Produksi"],
        [MERAH, HIJAU],
        ["Restan (kg)", "Produksi (kg)"],
    ):
        ax.fill_between(df["Waktu"], df[col], color=color, alpha=0.45)
        ax.plot(df["Waktu"], df[col], color=color, lw=1)
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_facecolor("#FAFAFA")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        ax.grid(alpha=0.25)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Restan per Bulan ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Restan per Bulan</div>',
                    unsafe_allow_html=True)
        monthly = df.groupby("Bulan").agg(
            rata_restan=("Restan","mean"),
            pct=("Restan_Binary", lambda x: x.mean()*100)
        )
        monthly.index = [BULAN_LABEL[i] for i in monthly.index]
        fig, ax1 = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#FAFAFA")
        x = np.arange(len(monthly))
        ax1.bar(x, monthly["rata_restan"], color=MERAH, alpha=0.7)
        ax1.set_xticks(x); ax1.set_xticklabels(monthly.index)
        ax1.set_ylabel("Rata-rata Restan (kg)", fontsize=8)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax1.set_facecolor("#FAFAFA")
        for sp in ["top","right"]: ax1.spines[sp].set_visible(False)
        ax2 = ax1.twinx()
        ax2.plot(x, monthly["pct"], color=BIRU, marker="o", lw=2)
        ax2.set_ylabel("% Hari Ada Restan", fontsize=8, color=BIRU)
        ax2.tick_params(axis="y", labelcolor=BIRU)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown('<div class="section-title">Restan per Hari dalam Seminggu</div>',
                    unsafe_allow_html=True)
        dow = df.groupby("DayOfWeek").agg(
            rata_restan=("Restan","mean"),
            pct=("Restan_Binary", lambda x: x.mean()*100)
        ).reindex(range(7))
        dow.index = HARI_LABEL
        colors_bar = [MERAH if v >= dow["rata_restan"].mean() else "#EF9A9A"
                      for v in dow["rata_restan"]]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#FAFAFA")
        ax.bar(dow.index, dow["rata_restan"], color=colors_bar, alpha=0.85)
        ax.axhline(dow["rata_restan"].mean(), color="grey", lw=1.2, ls="--")
        ax.set_xticklabels(HARI_LABEL, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Rata-rata Restan (kg)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_facecolor("#FAFAFA")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — EKSPLORASI DATA
# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "🔍 Eksplorasi Data":
    st.title("🔍 Eksplorasi Data Interaktif")

    # ── Filter sidebar ────────────────────────────────────────────────────────
    st.sidebar.markdown("### Filter Data")
    bulan_pilih = st.sidebar.multiselect(
        "Bulan", options=sorted(df["Bulan"].unique()),
        default=sorted(df["Bulan"].unique()),
        format_func=lambda x: BULAN_LABEL[x],
    )
    hari_pilih = st.sidebar.multiselect(
        "Hari", options=list(range(7)), default=list(range(7)),
        format_func=lambda x: HARI_LABEL[x],
    )
    df_fil = df[df["Bulan"].isin(bulan_pilih) & df["DayOfWeek"].isin(hari_pilih)]
    st.caption(f"Menampilkan **{len(df_fil)}** dari {len(df)} hari kerja")

    # ── Heatmap Korelasi ──────────────────────────────────────────────────────
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown('<div class="section-title">Heatmap Korelasi Antar Variabel</div>',
                    unsafe_allow_html=True)
        corr_cols = ["Restan","Produksi","Jumlah Pemanen","% Langsir",
                     "Curah Hujan","Jumlah Trip/Hari","Rata-rata Ritase/Hari",
                     "Jam Timbang Pertama","Bulan"]
        corr = df_fil[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        fig.patch.set_facecolor("#FAFAFA")
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title("Korelasi Pearson", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Scatter Plot Interaktif</div>',
                    unsafe_allow_html=True)
        num_cols_scatter = ["Restan","Produksi","Jumlah Pemanen","% Langsir",
                            "Curah Hujan","Jumlah Trip/Hari","Jam Timbang Pertama"]
        x_var = st.selectbox("Sumbu X", num_cols_scatter, index=2)
        y_var = st.selectbox("Sumbu Y", num_cols_scatter, index=0)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor("#FAFAFA")
        colors_sc = df_fil["Restan_Binary"].map({0: HIJAU, 1: MERAH})
        ax.scatter(df_fil[x_var], df_fil[y_var], c=colors_sc, alpha=0.65, s=40, edgecolors="none")
        ax.set_xlabel(x_var, fontsize=9)
        ax.set_ylabel(y_var, fontsize=9)
        ax.set_facecolor("#FAFAFA")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        legend_handles = [
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=HIJAU, markersize=9, label="Tanpa Restan"),
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=MERAH, markersize=9, label="Ada Restan"),
        ]
        ax.legend(handles=legend_handles, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Komparasi kondisi ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Distribusi Variabel: Ada vs Tanpa Restan</div>',
                unsafe_allow_html=True)
    var_pilih = st.selectbox("Pilih variabel", 
                             ["Jumlah Pemanen","% Langsir","Curah Hujan",
                              "Jumlah Trip/Hari","Rata-rata Ritase/Hari","Jam Timbang Pertama"])
    no_r = df_fil[df_fil["Restan"] == 0][var_pilih].dropna()
    ya_r = df_fil[df_fil["Restan"] >  0][var_pilih].dropna()

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#FAFAFA")
    ax.hist(no_r, bins=18, color=HIJAU, alpha=0.6, label=f"Tanpa Restan (n={len(no_r)})", density=True)
    ax.hist(ya_r, bins=18, color=MERAH, alpha=0.6, label=f"Ada Restan (n={len(ya_r)})", density=True)
    ax.axvline(no_r.mean(), color=HIJAU, lw=2, ls="--")
    ax.axvline(ya_r.mean(), color=MERAH, lw=2, ls="--")
    ax.set_xlabel(var_pilih)
    ax.set_ylabel("Densitas")
    ax.set_facecolor("#FAFAFA")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Tabel data ────────────────────────────────────────────────────────────
    with st.expander("📋 Lihat Tabel Data"):
        st.dataframe(
            df_fil[["Waktu","Restan","Restan_Binary","Produksi","Jumlah Pemanen",
                    "% Langsir","Curah Hujan","Jumlah Trip/Hari","Jam Timbang Pertama"]
            ].rename(columns={"Restan_Binary": "Ada Restan?"})
            .style.format({"Restan": "{:,.0f}", "Produksi": "{:,.0f}"})
            .applymap(lambda v: "background-color:#FFEBEE" if v == 1 else
                               "background-color:#E8F5E9" if v == 0 else "",
                      subset=["Ada Restan?"]),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 3 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "🤖 Prediksi Restan":
    st.title("🤖 Prediksi Restan Hari Ini")
    st.caption("Masukkan kondisi hari ini untuk memprediksi apakah akan terjadi restan")

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown("### Input Kondisi Hari Ini")
        jumlah_pemanen   = st.slider("Jumlah Pemanen (orang)",
                                     int(df["Jumlah Pemanen"].min()),
                                     int(df["Jumlah Pemanen"].max()),
                                     int(df["Jumlah Pemanen"].median()))
        pct_langsir      = st.slider("% Langsir",
                                     float(df["% Langsir"].min()),
                                     float(df["% Langsir"].max()),
                                     float(df["% Langsir"].median()), step=0.1)
        curah_hujan      = st.slider("Curah Hujan (mm)", 0.0,
                                     float(df["Curah Hujan"].max()),
                                     float(df["Curah Hujan"].median()), step=0.1)
        jumlah_trip      = st.slider("Jumlah Trip/Hari",
                                     int(df["Jumlah Trip/Hari"].min()),
                                     int(df["Jumlah Trip/Hari"].max()),
                                     int(df["Jumlah Trip/Hari"].median()))
        rata_ritase      = st.slider("Rata-rata Ritase/Hari",
                                     float(df["Rata-rata Ritase/Hari"].min()),
                                     float(df["Rata-rata Ritase/Hari"].max()),
                                     float(df["Rata-rata Ritase/Hari"].median()), step=0.01)
        jam_timbang      = st.slider("Jam Timbang Pertama (format desimal, misal 8.5 = 08:30)",
                                     0.0, 23.99,
                                     float(df["Jam Timbang Pertama"].median()), step=0.01)
        bulan            = st.selectbox("Bulan", options=list(BULAN_LABEL.keys()),
                                        format_func=lambda x: BULAN_LABEL[x],
                                        index=0)
        hari             = st.selectbox("Hari", options=list(range(7)),
                                        format_func=lambda x: HARI_LABEL[x])

    with col_result:
        st.markdown("### Hasil Prediksi")
        X_input = pd.DataFrame([{
            "Jumlah Pemanen":         jumlah_pemanen,
            "% Langsir":              pct_langsir,
            "Curah Hujan":            curah_hujan,
            "Jumlah Trip/Hari":       jumlah_trip,
            "Rata-rata Ritase/Hari":  rata_ritase,
            "Jam Timbang Pertama":    jam_timbang,
            "Bulan":                  bulan,
            "DayOfWeek":              hari,
        }])

        pred_rf = rf_model.predict(X_input)[0]
        prob_rf = rf_model.predict_proba(X_input)[0][1]
        pred_dt = dt_model.predict(X_input)[0]
        prob_dt = dt_model.predict_proba(X_input)[0][1]

        # Consensus
        verdict = pred_rf  # gunakan RF sebagai model utama

        if verdict == 1:
            st.markdown(f"""
            <div class="pred-box pred-restan">
                <div style="font-size:44px">⚠️</div>
                <div style="font-size:20px; font-weight:700; color:{MERAH}; margin:6px 0">
                    BERPOTENSI ADA RESTAN
                </div>
                <div style="font-size:13px; color:#555">
                    Probabilitas: <b>{prob_rf*100:.1f}%</b> (Random Forest)
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-box pred-aman">
                <div style="font-size:44px">✅</div>
                <div style="font-size:20px; font-weight:700; color:{HIJAU}; margin:6px 0">
                    AMAN — TIDAK ADA RESTAN
                </div>
                <div style="font-size:13px; color:#555">
                    Probabilitas restan: <b>{prob_rf*100:.1f}%</b> (Random Forest)
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Random Forest", f"{prob_rf*100:.1f}% restan",
                      delta="⚠️ Berisiko" if pred_rf else "✅ Aman",
                      delta_color="inverse")
        col_m2.metric("Decision Tree", f"{prob_dt*100:.1f}% restan",
                      delta="⚠️ Berisiko" if pred_dt else "✅ Aman",
                      delta_color="inverse")

        # Gauge-style bar
        st.markdown(f"**Tingkat Risiko (RF)**")
        st.progress(min(int(prob_rf * 100), 100))

        # Visualisasi Decision Tree path
        with st.expander("🌳 Lihat Visualisasi Decision Tree"):
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor("#FAFAFA")
            plot_tree(dt_model, feature_names=FEATURES,
                      class_names=["Tanpa Restan", "Ada Restan"],
                      filled=True, rounded=True, fontsize=8, ax=ax,
                      impurity=False, proportion=True)
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 4 — REKOMENDASI
# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "💡 Rekomendasi":
    st.title("💡 Rekomendasi Operasional")
    st.caption("Insight berbasis data untuk meminimalkan kejadian restan")

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("### Variabel Paling Berpengaruh terhadap Restan")
    importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    feat_labels = {
        "Jumlah Pemanen":        "Jumlah Pemanen",
        "% Langsir":             "% Langsir",
        "Curah Hujan":           "Curah Hujan",
        "Jumlah Trip/Hari":      "Jumlah Trip/Hari",
        "Rata-rata Ritase/Hari": "Rata-rata Ritase/Hari",
        "Jam Timbang Pertama":   "Jam Timbang Pertama",
        "Bulan":                 "Bulan",
        "DayOfWeek":             "Hari dalam Seminggu",
    }
    colors_imp = [MERAH if v >= importances.median() else "#EF9A9A" for v in importances]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#FAFAFA")
    bars = ax.barh([feat_labels[f] for f in importances.index],
                   importances.values, color=colors_imp, edgecolor="white")
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=9)
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, importances.max() * 1.25)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Tabel perbandingan kondisi ─────────────────────────────────────────────
    st.markdown("### Rata-rata Kondisi: Ada vs Tanpa Restan")
    no_r = df[df["Restan"] == 0]
    ya_r = df[df["Restan"] >  0]
    compare_cols = ["Jumlah Pemanen", "% Langsir", "Curah Hujan",
                    "Jumlah Trip/Hari", "Rata-rata Ritase/Hari", "Jam Timbang Pertama"]
    tabel = pd.DataFrame({
        "Variabel":         compare_cols,
        "Tanpa Restan":     [round(no_r[c].mean(), 2) for c in compare_cols],
        "Ada Restan":       [round(ya_r[c].mean(), 2) for c in compare_cols],
        "Selisih":          [round(ya_r[c].mean() - no_r[c].mean(), 2) for c in compare_cols],
    })

    def color_selisih(val):
        if val > 0: return "color: #C62828; font-weight: bold"
        elif val < 0: return "color: #2E7D32; font-weight: bold"
        return ""

    st.dataframe(
        tabel.style.applymap(color_selisih, subset=["Selisih"]),
        use_container_width=True, hide_index=True,
    )

    # ── Rekomendasi Operasional ────────────────────────────────────────────────
    st.markdown("### Rekomendasi Operasional")

    rekomendasi = [
        {
            "icon": "👷",
            "judul": "Optimalkan Jumlah Pemanen",
            "isi": (
                f"Rata-rata pemanen saat **ada restan** justru lebih tinggi "
                f"({ya_r['Jumlah Pemanen'].mean():.0f} orang) dibanding tanpa restan "
                f"({no_r['Jumlah Pemanen'].mean():.0f} orang). "
                "Ini mengindikasikan bahwa **kuantitas pemanen bukan satu-satunya faktor** — "
                "efisiensi kerja dan kesiapan armada pengangkutan lebih krusial."
            ),
        },
        {
            "icon": "🚛",
            "judul": "Tingkatkan % Langsir & Ritase",
            "isi": (
                f"Hari tanpa restan memiliki rata-rata % Langsir lebih rendah "
                f"({no_r['% Langsir'].mean():.1f}%) dibanding hari ada restan "
                f"({ya_r['% Langsir'].mean():.1f}%). Kondisi ini perlu ditelaah lebih "
                "lanjut apakah terkait dengan rute pengangkutan atau kapasitas kendaraan. "
                "Pastikan ritase pengangkutan mencukupi sejak awal hari."
            ),
        },
        {
            "icon": "⏰",
            "judul": "Percepat Jam Timbang Pertama",
            "isi": (
                f"Hari tanpa restan memiliki rata-rata jam timbang lebih siang "
                f"({no_r['Jam Timbang Pertama'].mean():.2f}) sedangkan hari ada restan "
                f"lebih pagi ({ya_r['Jam Timbang Pertama'].mean():.2f}). "
                "Ini menarik — artinya permulaan aktivitas yang sangat dini belum tentu "
                "menjamin nihil restan. Fokus pada **kesiapan armada dan koordinasi** "
                "sebelum aktivitas dimulai."
            ),
        },
        {
            "icon": "📅",
            "judul": "Waspadai Periode April–Juni",
            "isi": (
                "Data menunjukkan lonjakan restan drastis mulai April (>92% hari ada restan). "
                "Rencanakan **penambahan kapasitas angkut dan pemanen cadangan** di semester awal "
                "tahun, terutama saat musim panen raya."
            ),
        },
        {
            "icon": "🌧️",
            "judul": "Monitor Curah Hujan Harian",
            "isi": (
                f"Curah hujan rata-rata lebih tinggi saat ada restan "
                f"({ya_r['Curah Hujan'].mean():.1f} mm vs {no_r['Curah Hujan'].mean():.1f} mm). "
                "Siapkan **prosedur kontingensi** untuk hari dengan prediksi hujan di atas 10 mm, "
                "termasuk penjadwalan ulang trip dan pengecekan kondisi jalan akses."
            ),
        },
    ]

    for rek in rekomendasi:
        st.markdown(f"""
        <div class="insight-box">
            <b>{rek['icon']} {rek['judul']}</b><br>
            <span style="font-size:14px">{rek['isi']}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "💡 **Catatan metodologi**: Model Random Forest mencapai akurasi 81% (F1 = 0.88) "
        "dengan 5-fold cross-validation pada 175 hari kerja. "
        "Rekomendasi di atas bersifat data-driven dan perlu dikonfirmasi dengan "
        "pengetahuan lapangan dari tim operasional.",
        icon="ℹ️",
    )
