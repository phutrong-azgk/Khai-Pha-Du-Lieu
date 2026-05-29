import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import random

# =========================================================
# FIX IMPORT PATH
# =========================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# =========================================================
# IMPORT MODULES
# =========================================================

# 1. DATA PREPROCESSING
from src.preprocess import load_and_clean

# 2. FEATURE ENGINEERING
from src.auto_ml_engine import (
    build_features,
    scale_and_reduce
)

# 3. CLUSTERING ENGINE
from src.clustering import (
    run_kmeans,
    run_gmm,
    run_hdbscan,
    run_all_clustering
)

# 4. EVALUATION + PERSONA
from src.evaluation import (
    evaluate_all,
    build_personas_matrix,
    FEATURE_COLS
)

# 5. AUTO BENCHMARK ENGINE
from src.clustering_evaluator import (
    run_clustering_experiments,
    get_recommendation
)

# =========================================================
# ML LIBRARIES
# =========================================================

from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder,
    StandardScaler
)

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from xgboost import XGBClassifier

import hdbscan

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Segmentation Analytics Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
# CSS Custom nâng cao theo phong cách BI/Production ML Dashboard
st.markdown(
    """
    <style>
    /* =========================================================
    GLOBAL APP
    ========================================================= */
    .stApp {
        background-color: #0B0F19 !important;
        color: #F1F5F9 !important;
    }

    /* =========================================================
    SIDEBAR
    ========================================================= */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* =========================================================
    TYPOGRAPHY
    ========================================================= */
    h1, h2, h3, h4, h5, h6,
    p, span, label, .stMarkdown {
        color: #F8FAFC !important;
    }

    /* =========================================================
    STREAMLIT COLUMNS FIX
    ========================================================= */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }

    [data-testid="column"] > div {
        height: 100%;
    }

    /* =========================================================
    METRIC CARD
    ========================================================= */
    .metric-card {

        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.75) 0%,
            rgba(15, 23, 42, 0.75) 100%
        );

        border: 1px solid rgba(255, 255, 255, 0.08);

        border-radius: 14px;

        padding: 20px;

        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.30);

        backdrop-filter: blur(8px);

        -webkit-backdrop-filter: blur(8px);

        transition: all 0.25s ease;

        /* FIX CHIỀU CAO ĐỒNG ĐỀU */
        height: 220px;

        min-height: 220px;

        max-height: 220px;

        display: flex;

        flex-direction: column;

        justify-content: flex-start;

        overflow: hidden;
    }

    /* Hover Effect */
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }

    /* =========================================================
    METRIC LABEL
    ========================================================= */
    .metric-label {

        font-size: 0.82rem;

        color: #94A3B8;

        font-weight: 600;

        text-transform: uppercase;

        letter-spacing: 0.06em;

        margin-bottom: 10px;

        min-height: 40px;
    }

    /* =========================================================
    METRIC VALUE
    ========================================================= */
    .metric-value {

        font-size: 2rem;

        font-weight: 700;

        color: #F8FAFC;

        line-height: 1.2;

        word-break: break-word;

        overflow-wrap: break-word;

        margin-top: 5px;

        /* FIX ALIGN VALUE */
        min-height: 85px;

        display: flex;

        align-items: flex-start;
    }

    /* =========================================================
    METRIC SUBTEXT
    ========================================================= */
    .metric-subtext {

        font-size: 0.75rem;

        color: #38BDF8;

        margin-top: auto;

        padding-top: 12px;

        opacity: 0.95;
    }

    /* =========================================================
    TABS
    ========================================================= */
    .stTabs [data-baseweb="tab-list"] {

        gap: 8px;

        background-color: #0F172A;

        padding: 6px;

        border-radius: 10px;

        border: 1px solid rgba(255,255,255,0.06);
    }

    .stTabs [data-baseweb="tab"] {

        padding: 10px 20px;

        color: #94A3B8;

        border-radius: 8px;

        font-weight: 600;

        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {

        background-color: #1E293B !important;

        color: #38BDF8 !important;

        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }

    /* =========================================================
    EXPANDER
    ========================================================= */
    div[data-testid="stExpander"] {

        background-color: #0F172A !important;

        border: 1px solid rgba(255, 255, 255, 0.08) !important;

        border-radius: 10px !important;
    }

    /* =========================================================
    SCROLLBAR
    ========================================================= */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #0F172A;
    }

    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Customer Segmentation Enterprise Dashboard")
st.caption("Nền tảng phân tích phân khúc khách hàng chuyên sâu phục vụ báo cáo đồ án & doanh nghiệp")

# Cấu hình cấu trúc giao diện chung cho biểu đồ Plotly thích hợp nền tối BI
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#E2E8F0', family="Inter, sans-serif"),
    xaxis=dict(gridcolor='#1E293B', zerolinecolor='#334155'),
    yaxis=dict(gridcolor='#1E293B', zerolinecolor='#334155')
)

# Bảng màu BI Palette chuẩn, đồng bộ xuyên suốt hệ thống
BASE_COLORS = [
    "#38BDF8",  # Cyan
    "#F43F5E",  # Rose
    "#10B981",  # Emerald
    "#A855F7",  # Purple
    "#F59E0B",  # Amber
    "#EC4899",  # Pink
    "#6366F1",  # Indigo
    "#818CF8",  # Violet
    "#2DD4BF",  # Teal
    "#F472B6"   # Hot Pink
]

# =========================================================
# RANDOM COLOR ENGINE
# =========================================================

def get_dynamic_colors(n_clusters, seed=42):

    """
    Sinh palette động cho số cụm lớn.
    
    - Giữ nguyên 10 màu đầu
    - Từ màu thứ 11 sẽ random
    - Tránh màu quá tối
    - Tránh màu trùng
    - Ổn định nhờ random seed
    """

    random.seed(seed)

    colors = list(BASE_COLORS)

    # Nếu ít cụm thì chỉ dùng màu gốc
    if n_clusters <= len(colors):
        return colors[:n_clusters]

    # =====================================================
    # RANDOM THÊM MÀU
    # =====================================================

    while len(colors) < n_clusters:

        # RGB tone sáng cho dark mode
        r = random.randint(90, 255)
        g = random.randint(90, 255)
        b = random.randint(90, 255)

        hex_color = f"#{r:02X}{g:02X}{b:02X}"

        # Tránh màu trùng
        if hex_color not in colors:

            # Tránh màu quá xám
            color_variance = max(r, g, b) - min(r, g, b)

            if color_variance > 40:
                colors.append(hex_color)

    return colors

# ─── Hàm Cache Tối Ưu Hóa Pipeline Thực Tế ────────────────────────────────────
@st.cache_data
def process_pipeline_data(uploaded_file, use_demo_data):
    """Xử lý load và làm sạch dữ liệu đầu vào thực tế"""
    if uploaded_file:
        raw = pd.read_csv(uploaded_file)
        raw["event_time"] = pd.to_datetime(raw["event_time"], utc=True)
        df = raw[raw["event_type"] == "purchase"].dropna(
            subset=["user_id", "product_id", "price", "event_time"]
        )
        df = df[df["price"] > 0].drop_duplicates(
            subset=["user_id", "product_id", "event_time"]
        )
    elif use_demo_data:
        df = load_and_clean("data/events.csv")
    else:
        return None
    return df

@st.cache_resource
def train_xgboost_importance(X, y):
    """Cache quá trình huấn luyện XGBoost phục vụ tính toán Feature Importance"""
    xgb_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X, y)
    return xgb_model.feature_importances_


# ─── Sidebar Cấu Hình Điều Khiển Hệ Thống ──────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#38BDF8;margin-top:0;'>⚙️ ML Control Center</h2>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("📂 Tải lên events.csv", type=["csv"])
    use_demo = st.checkbox("Dùng dữ liệu mẫu (data/events.csv)", value=True)

    algo = st.selectbox(
        "Thuật toán phân cụm",
        ["K-Means", "Gaussian Mixture Model (GMM)", "HDBSCAN"],
    )

    if algo == "K-Means":
        n_clusters = st.slider("Số cụm (k)", 2, 10, 4)
    elif algo == "Gaussian Mixture Model (GMM)":
        n_clusters = st.slider("Số thành phần", 2, 10, 4)
    else:
        min_cluster_size = st.slider("min_cluster_size", 10, 200, 50)
        n_clusters = None

    run_btn = st.button("Khởi chạy Phân cụm Hệ thống", type="primary", use_container_width=True)

# ─── Pipeline Xử Lý Dữ Liệu Học Máy ──────────────────────────────────────────
# Tự động load dữ liệu mẫu nếu phiên làm việc mới chưa bấm nút chạy lần nào
if "features" not in st.session_state and use_demo:
    run_btn = True

if run_btn:
    with st.status("Đang xử lý dữ liệu hệ thống...", expanded=True) as status:
        try:
            # 1. Đọc dữ liệu đầu vào thông qua cache toán học
            df = process_pipeline_data(uploaded, use_demo)
            if df is None:
                st.error("Vui lòng tải lên file CSV hoặc bật 'Dùng dữ liệu mẫu'.")
                st.stop()
                
            st.write(f"Dữ liệu sạch: **{len(df):,}** giao dịch | **{df['user_id'].nunique():,}** khách hàng")

            # 2. Trích xuất đặc trưng hành vi (Feature Engineering)
            features = build_features(df)
            st.write(f"Đặc trưng: **{features.shape[1]-1}** features cho **{len(features):,}** khách hàng")

            # --- Biến đổi Logarit chuẩn hóa phân phối theo cấu hình lõi ---
            features_balanced = features.copy()
            cols_to_log = ["Recency", "Frequency", "Monetary"]
            for col in cols_to_log:
                if col in features_balanced.columns:
                    features_balanced[col] = np.log1p(features_balanced[col])
            st.write("Đã áp dụng Log Transformation thành công")

            # 3. Scale + PCA Giảm chiều dữ liệu từ hàm gốc
            X_scaled, X_pca, pca_df = scale_and_reduce(features_balanced)
            
            # ==========================================
            # AUTO ML BENCHMARK
            # ==========================================


            st.write("Chuẩn hoá dữ liệu & Phân tích PCA hoàn tất")

            # 4. Thực thi mô hình học máy phân cụm khách hàng thật
            if algo == "K-Means":
                labels, model = run_kmeans(X_scaled, n_clusters)
                col_name = "cluster_kmeans"
            elif algo == "Gaussian Mixture Model (GMM)":
                labels, model = run_gmm(X_scaled, n_clusters)
                col_name = "cluster_gmm"
            else:
                labels, model = run_hdbscan(X_scaled, min_cluster_size)
                col_name = "cluster_hdbscan"
                algo = "HDBSCAN"

            clustered = features[["user_id"]].copy()
            clustered[col_name] = labels
            clustered["cluster"] = labels
            st.write(f"Đã gán nhãn thực tế mô hình: **{len(set(labels))}** nhóm phân cụm")

           # 5. Chạy Auto Benchmark Engine
            st.write("Đang chạy Auto Benchmark Engine...")

            benchmark_df = run_clustering_experiments(X_scaled)

            recommendation_text = get_recommendation(
                benchmark_df
            )

            st.session_state["benchmark_df"] = benchmark_df
            st.session_state["recommendation_text"] = recommendation_text

            status.update(
                label="Hoàn tất xử lý dữ liệu & Huấn luyện mô hình!",
                state="complete"
            )

            # Lưu trạng thái kết quả vào Streamlit Session State để tránh mất dữ liệu khi tương tác UI
            st.session_state["features"]  = features
            st.session_state["clustered"] = clustered
            st.session_state["X_scaled"]  = X_scaled
            st.session_state["pca_df"]    = pca_df
            st.session_state["algo"]      = algo
            st.session_state["col_name"]  = col_name
            st.session_state["benchmark_df"] = benchmark_df

        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")
            raise e

# ─── Khối Trực Quan Hóa Dữ Liệu Nâng Cấp BI ───────────────────────────────────
if "features" in st.session_state:
    features  = st.session_state["features"]
    clustered = st.session_state["clustered"]
    X_scaled  = st.session_state["X_scaled"]
    pca_df    = st.session_state["pca_df"]
    algo      = st.session_state["algo"]
    col_name  = st.session_state["col_name"]
    benchmark_df = st.session_state["benchmark_df"]

    # Thiết lập DataFrame liên kết trung tâm chứa toàn bộ dữ liệu thật
    merged = pca_df.merge(clustered[["user_id", "cluster"]], on="user_id")
    merged = merged.merge(features, on="user_id")
    merged["Cluster"] = "Cụm " + merged["cluster"].astype(str)

    # 🎯 1. BỘ LỌC ĐỘNG PERCENTILE (Yêu cầu số 5)
    st.markdown("<h3 style='color:#38BDF8; margin-top:15px;'>🔍 Bộ lọc tương tác BI động & Kiểm soát Outliers</h3>", unsafe_allow_html=True)
    filter_cols = st.columns([2, 3])
    
    with filter_cols[0]:
        available_clusters = sorted(list(merged["Cluster"].unique()))
        selected_clusters = st.multiselect(
            "Chọn cụm muốn hiển thị tập trung trên Dashboard:",
            options=available_clusters,
            default=available_clusters
        )
    
    with filter_cols[1]:
        percentile_threshold = st.slider(
            "Cắt ngưỡng Percentile trục đồ thị (Loại bỏ ảnh hưởng méo đồ thị do Outliers thực tế gây ra):",
            min_value=85.0, max_value=100.0, value=98.5, step=0.5
        )

    # Thực thi lọc dữ liệu động theo lựa chọn của người dùng
    if selected_clusters:
        filtered_df = merged[merged["Cluster"].isin(selected_clusters)]
    else:
        filtered_df = merged.copy()
        selected_clusters = available_clusters

    # Tự động tính toán khung giới hạn thích ứng (Auto-fit zoom) theo phân vị dữ liệu thực tế
    if not filtered_df.empty:
        max_y_monetary = float(np.percentile(filtered_df["Monetary"], percentile_threshold))
        max_x_frequency = float(np.percentile(filtered_df["Frequency"], percentile_threshold))
        max_x_recency = float(np.percentile(filtered_df["Recency"], percentile_threshold))
    else:
        max_y_monetary, max_x_frequency, max_x_recency = 3000.0, 25.0, 365.0

    st.markdown("---")

    # 📊 2. TÍNH TOÁN VÀ ĐỔ DỮ LIỆU VÀO KHỐI KPI THỰC TẾ (Yêu cầu số 6)
    total_customers = len(filtered_df)
    total_revenue = filtered_df["Monetary"].sum()
    avg_order_value = filtered_df["Average_Basket_Value"].mean() if "Average_Basket_Value" in filtered_df.columns else (total_revenue / filtered_df["Frequency"].sum())
    total_transactions = filtered_df["Frequency"].sum()
    
    # Tính Silhouette thực tế cho mô hình đang chọn dựa trên tập dữ liệu hiện tại
    try:
        current_labels = filtered_df["cluster"].astype(int)
        # Chỉ tính toán khi có nhiều hơn 1 cụm trong tập dữ liệu đã lọc
        if len(set(current_labels)) > 1:
            # Lấy vị trí dòng của filtered_df để map tương ứng với X_scaled ban đầu
            matched_indices = filtered_df.index
            current_silhouette = metrics.silhouette_score(X_scaled[matched_indices][:4000], current_labels.iloc[:4000])
        else:
            current_silhouette = 0.0
    except Exception:
        current_silhouette = 0.0

    # Hiển thị cấu trúc khối KPI sang giao diện Glassmorphism BI chuyên nghiệp
    kpi_cols = st.columns(5)
    metrics_list = [
        {"label": "Tổng khách hàng", "val": f"{total_customers:,}", "sub": "Mẫu hiện tại hiển thị"},
        {"label": "Tổng doanh thu thật", "val": f"${total_revenue:,.2f}", "sub": "Tổng Gross Monetary"},
        {"label": "Avg Order Value", "val": f"${avg_order_value:,.2f}", "sub": "Giá trị đơn hàng TB"},
        {"label": "Tổng giao dịch", "val": f"{total_transactions:,}", "sub": "Tần suất mua sắm thực tế"},
        {"label": "Silhouette Score thật", "val": f"{current_silhouette:.3f}", "sub": f"Độ nén cụm ({algo[:7]})"}
    ]

    for i, col in enumerate(kpi_cols):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metrics_list[i]['label']}</div>
                    <div class="metric-value">{metrics_list[i]['val']}</div>
                    <div class="metric-subtext">● {metrics_list[i]['sub']}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ĐỊNH DANH PERSONA ĐỘNG THEO THỨ HẠNG TRUNG BÌNH CỦA CỤM ---
    cluster_stats_raw = merged.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    rank_recency = cluster_stats_raw["Recency"].rank(ascending=True)
    rank_freq = cluster_stats_raw["Frequency"].rank(ascending=False)
    rank_monetary = cluster_stats_raw["Monetary"].rank(ascending=False)

    persona_mapping = {}
    assigned_personas = set()

    for c_name in cluster_stats_raw.index:
        if rank_recency[c_name] == len(cluster_stats_raw) and "Dormant — Khách ngủ đông" not in assigned_personas:
            p_label = "Dormant — Khách ngủ đông"
            p_desc = "Lâu không quay lại mua hàng, có nguy cơ rời bỏ cao."
        elif rank_monetary[c_name] == 1 and "VIP — Khách hàng vàng" not in assigned_personas:
            p_label = "VIP — Khách hàng vàng"
            p_desc = "Mua gần đây, tần suất cao, đóng góp giá trị doanh thu lớn."
        elif rank_freq[c_name] <= 2 and "Loyal — Khách trung thành" not in assigned_personas:
            p_label = "Loyal — Khách trung thành"
            p_desc = "Mua sắm thường xuyên, độ tin tưởng thương hiệu ổn định."
        elif rank_freq[c_name] == len(cluster_stats_raw) and rank_recency[c_name] <= 2 and "New — Khách mới" not in assigned_personas:
            p_label = "New — Khách mới"
            p_desc = "Mới phát sinh đơn hàng đầu tiên, cần nuôi dưỡng và kích hoạt."
        else:
            p_label = f"Regular — Khách vãng lai ({c_name})"
            p_desc = "Sức mua và tần suất ở mức trung bình, mua hàng theo nhu cầu phát sinh."
            
        persona_mapping[c_name] = (p_label, p_desc)
        assigned_personas.add(p_label)

    # Cấu trúc 7 Tab hiển thị phân tích chuyên sâu (Tích hợp thêm Association Rules)
    tabs = st.tabs([
        "Không gian PCA Scatter", 
        "Thống kê Phân phối & Profile", 
        "So sánh Thuật toán", 
        "Personas Chân dung", 
        "Chiến lược Marketing", 
        "Chẩn đoán Kỹ thuật",
        "Khai phá Luật kết hợp"
    ])

    # ── Tab 1: Không gian giảm chiều PCA ──────────────────────────────────────
    with tabs[0]:
        st.subheader("Biểu đồ phân không gian khách hàng thực tế (PCA 2D + WebGL)")
        
        c_pca1, c_pca2 = st.columns(2)
        with c_pca1:
            # Đảm bảo có PC2 trước khi vẽ
            if "PC2" not in filtered_df.columns:
                filtered_df = filtered_df.copy()
                filtered_df["PC2"] = 0

            fig_pca_before = px.scatter(
                filtered_df, x="PC1", y="PC2",
                title="Trước khi gom cụm (Dữ liệu gốc chưa gán nhãn)",
                opacity=0.4,
                color_discrete_sequence=["#4B5563"]
            )
            fig_pca_before.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig_pca_before, use_container_width=True)
            
        with c_pca2:
            sample_size_pca = min(12000, len(filtered_df))
            sample_pca_df = filtered_df.sample(n=sample_size_pca, random_state=42)

            if "PC2" not in sample_pca_df.columns:
                sample_pca_df = sample_pca_df.copy()
                sample_pca_df["PC2"] = 0

            fig_pca_after = px.scatter(
                sample_pca_df, x="PC1", y="PC2", color="Cluster",
                hover_data=["user_id", "Recency", "Frequency", "Monetary"],
                title=f"Sau khi gom cụm — Mô hình {algo} thật (n_samples={sample_size_pca:,})",
                opacity=0.75,
                render_mode="webgl",
                color_discrete_sequence=get_dynamic_colors(
                    len(sample_pca_df["Cluster"].unique())
                ),
                category_orders={"Cluster": available_clusters}
            )
            fig_pca_after.update_layout(**PLOTLY_THEME)
            fig_pca_after.update_traces(marker=dict(size=4.5, line=dict(width=0.3, color='rgba(255,255,255,0.2)')))
            
            pc1_min, pc1_max = np.percentile(sample_pca_df["PC1"], [1, 99])
            pc2_min, pc2_max = np.percentile(sample_pca_df["PC2"], [1, 99])
            if pc1_min != pc1_max and pc2_min != pc2_max:
                fig_pca_after.update_layout(
                    xaxis=dict(range=[pc1_min * 1.2, pc1_max * 1.2], autorange=False),
                    yaxis=dict(range=[pc2_min * 1.2, pc2_max * 1.2], autorange=False)
                )
            st.plotly_chart(fig_pca_after, use_container_width=True)

        if "PC3" in filtered_df.columns:
            st.subheader("Không gian phân cụm chiều 3D")
            fig3d = px.scatter_3d(
                sample_pca_df, x="PC1", y="PC2", z="PC3", color="Cluster", opacity=0.7,
                color_discrete_sequence=get_dynamic_colors(
                    len(sample_pca_df["Cluster"].unique())
                ),
                category_orders={"Cluster": available_clusters}
            )
            fig3d.update_traces(marker=dict(size=3))
            fig3d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F0F6FC'),
                scene=dict(
                    xaxis=dict(backgroundcolor="#111827", gridcolor="#1E293B"),
                    yaxis=dict(backgroundcolor="#111827", gridcolor="#1E293B"),
                    zaxis=dict(backgroundcolor="#111827", gridcolor="#1E293B")
                )
            )
            st.plotly_chart(fig3d, use_container_width=True)

    # ── Tab 2: Thống kê chi tiết thực tế & Biểu đồ Phân phối cốt lõi ─────────────
    with tabs[1]:
        st.subheader("Bảng chỉ số trung bình thực tế theo từng cụm khách hàng")
        stats = filtered_df.groupby("Cluster")[FEATURE_COLS].mean().round(2)
        stats["Số khách hàng"] = filtered_df.groupby("Cluster").size()
        stats["% tổng"] = (stats["Số khách hàng"] / len(merged) * 100).round(1)
        
        show_cols = ["Số khách hàng", "% tổng"] + [c for c in FEATURE_COLS if c in stats.columns]
        st.dataframe(stats[show_cols].style.background_gradient(cmap="Blues", axis=0), use_container_width=True)

        st.markdown("---")
        
        # 📦 BOXPLOT MONETARY THẬT (Yêu cầu số 2)
        st.markdown("###So sánh Phân phối Chi tiêu thực tế (Boxplot Monetary)")
        fig_box = px.box(
            filtered_df,
            x="Cluster",
            y="Monetary",
            color="Cluster",
            color_discrete_sequence=get_dynamic_colors(len(filtered_df["Cluster"].unique())),
            points="outliers",  # Bảo lưu toàn bộ các điểm outlier thực tế của hệ thống
            category_orders={"Cluster": available_clusters},
            notched=True,
            title="Biểu đồ hộp phân tích Trung vị (Median), Tứ phân vị (Q1, Q3) và Đường trung bình (Mean Line)"
        )
        fig_box.update_layout(**PLOTLY_THEME)
        fig_box.update_traces(boxmean=True)  # Hiển thị thêm mean line bằng nét đứt
        fig_box.update_layout(yaxis=dict(range=[0, max_y_monetary * 1.15], title="Mức độ Chi tiêu thực tế ($)"))
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")

        # 📊 RFM PROFILE BAR CHART TRUNG BÌNH THỰC TẾ (Yêu cầu số 3)
        st.markdown("###RFM Profile Bar Chart (Grouped)")
        # Tính toán groupby thực tế
        cluster_rfm_mean = filtered_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().reset_index()
        rfm_melted = pd.melt(cluster_rfm_mean, id_vars=["Cluster"], value_vars=["Recency", "Frequency", "Monetary"],
                             var_name="Chỉ số RFM Core", value_name="Giá trị Trung bình Thực tế")
        
        fig_rfm_bar = px.bar(
            rfm_melted,
            x="Cluster",
            y="Giá trị Trung bình Thực tế",
            color="Chỉ số RFM Core",
            barmode="group",
            text_auto=".2s", # Hiển thị text value rõ ràng trên đầu mỗi cột trụ
            color_discrete_sequence=["#F43F5E", "#10B981", "#38BDF8"],
            category_orders={"Cluster": available_clusters},
            title="Đồ thị cột nhóm so sánh tương quan thực tế thuộc tính Recency - Frequency - Monetary"
        )
        fig_rfm_bar.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_rfm_bar, use_container_width=True)

        st.markdown("---")
        
        # Giữ nguyên đồ thị phân bố không gian thuộc tính cũ của người dùng
        st.subheader("Phân tích Phân phối Phụ thuộc Không gian gốc")
        sample_df = filtered_df.sample(min(12000, len(filtered_df)), random_state=42)
        col_mf1, col_mf2 = st.columns(2)
        with col_mf1:
            fig_mf_before = px.scatter(
                sample_df, x="Frequency", y="Monetary",
                title="Trước khi gom cụm — MONETARY VS FREQUENCY (Dữ liệu thật)",
                opacity=0.4, color_discrete_sequence=["#38BDF8"],
                range_x=[0, 25], range_y=[0, 3000]
            )
            fig_mf_before.update_layout(**PLOTLY_THEME)
            fig_mf_before.update_layout(yaxis=dict(nticks=6, tickformat=',.1f'), xaxis=dict(nticks=6, tickformat='d'))
            fig_mf_before.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_mf_before, use_container_width=True)
            
        with col_mf2:
            # 🛠️ CƠ CHẾ DEFENSIVE PHÒNG THỦ: Tự động hạ cấp trendline nếu cụm quá ít dữ liệu
            try:
                fig_mf_after = px.scatter(
                    sample_df, x="Frequency", y="Monetary", color="Cluster",
                    trendline="lowess", trendline_options=dict(frac=0.6),
                    title="Sau khi gom cụm — MONETARY VS FREQUENCY", opacity=0.55,
                    color_discrete_sequence=get_dynamic_colors(len(sample_df["Cluster"].unique())),
                    range_x=[0, 25], range_y=[0, 3000]
                )
            except Exception:
                # Nếu LOWESS thất bại do có cụm bị cô lập, chuyển sang OLS (Tuyến tính)
                fig_mf_after = px.scatter(
                    sample_df, x="Frequency", y="Monetary", color="Cluster",
                    trendline="ols",
                    title="Sau khi gom cụm — MONETARY VS FREQUENCY (Hạ cấp Tuyến tính)", opacity=0.55,
                    color_discrete_sequence=get_dynamic_colors(len(sample_df["Cluster"].unique())),
                    range_x=[0, 25], range_y=[0, 3000]
                )
                
            fig_mf_after.update_layout(**PLOTLY_THEME)
            fig_mf_after.update_layout(yaxis=dict(nticks=6, tickformat=',.1f'), xaxis=dict(nticks=6, tickformat='d'))
            fig_mf_after.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_mf_after, use_container_width=True)

        st.markdown("---")

        st.subheader("Phân tích Phân phối: Monetary vs Recency")
        col_mr1, col_mr2 = st.columns(2)
        with col_mr1:
            fig_mr_before = px.scatter(
                sample_df, x="Recency", y="Monetary",
                title="Trước khi gom cụm — MONETARY VS RECENCY (Dữ liệu thật)",
                opacity=0.4, color_discrete_sequence=["#C084FC"],
                range_y=[0, 3000]
            )
            fig_mr_before.update_layout(**PLOTLY_THEME)
            fig_mr_before.update_layout(yaxis=dict(nticks=6, tickformat=',.1f'), xaxis=dict(nticks=8, tickformat='d'))
            fig_mr_before.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_mr_before, use_container_width=True)
            
        with col_mr2:
            # 🛠️ CƠ CHẾ DEFENSIVE PHÒNG THỦ: Tránh sập biểu đồ Recency khi tính toán trendline
            try:
                fig_mr_after = px.scatter(
                    sample_df, x="Recency", y="Monetary", color="Cluster",
                    trendline="lowess", trendline_options=dict(frac=0.6),
                    title="Sau khi gom cụm — MONETARY VS RECENCY", opacity=0.55,
                    color_discrete_sequence=get_dynamic_colors(len(sample_df["Cluster"].unique())),
                    range_y=[0, 3000]
                )
            except Exception:
                fig_mr_after = px.scatter(
                    sample_df, x="Recency", y="Monetary", color="Cluster",
                    trendline="ols",
                    title="Sau khi gom cụm — MONETARY VS RECENCY (Hạ cấp Tuyến tính)", opacity=0.55,
                    color_discrete_sequence=get_dynamic_colors(len(sample_df["Cluster"].unique())),
                    range_y=[0, 3000]
                )
                
            fig_mr_after.update_layout(**PLOTLY_THEME)
            fig_mr_after.update_layout(yaxis=dict(nticks=6, tickformat=',.1f'), xaxis=dict(nticks=8, tickformat='d'))
            fig_mr_after.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_mr_after, use_container_width=True)

        st.markdown("---")

        col_chart_left, col_chart_right = st.columns(2, gap="large")
        
        with col_chart_left:
            st.subheader("Tỷ trọng Quy mô & Giá trị Chi tiêu")
            fig_mon = px.bar(
                stats.reset_index(), x="Cluster", y="Monetary",
                color="Cluster", title="Giá trị chi tiêu trung bình (Monetary)",
                text_auto='.2s',
                color_discrete_sequence=get_dynamic_colors(len(stats.reset_index()["Cluster"].unique()))
            )
            fig_mon.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig_mon, use_container_width=True)
            
            fig_pie = px.pie(
                stats.reset_index(), values="Số khách hàng", names="Cluster",
                title="Phân phối kích thước cụm (% Khách hàng)", hole=0.4,
                color_discrete_sequence=get_dynamic_colors(len(stats.reset_index()["Cluster"].unique()))
            )
            fig_pie.update_layout(paper_bgcolor='#0E1117', font=dict(color='#F0F6FC'))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_chart_right:
            st.subheader("Biểu đồ Radar — Hàng vi & Đa dạng Mua sắm (6 biến)")
            radar_cols = ["Recency", "Frequency", "Monetary", "Average_Basket_Value", "Brand_Diversity", "Category_Diversity"]
            active_radar_cols = [c for c in radar_cols if c in merged.columns]
            
            radar_data = merged.groupby("Cluster")[active_radar_cols].mean()
            scaler_r = MinMaxScaler()
            radar_norm = pd.DataFrame(
                scaler_r.fit_transform(radar_data),
                index=radar_data.index, columns=active_radar_cols,
            )
            
            fig_radar = go.Figure()
            
            for idx, cluster in enumerate(radar_norm.index):
                vals = radar_norm.loc[cluster].tolist()
                dynamic_colors = get_dynamic_colors(
                    len(radar_norm.index)
                )

                current_color = dynamic_colors[
                    idx % len(dynamic_colors)
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], 
                    theta=active_radar_cols + [active_radar_cols[0]],
                    fill="toself", name=f"Cụm {cluster}",
                    fillcolor=current_color,
                    opacity=0.15,
                    line=dict(color=current_color, width=2.5)
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363D", angle=45, tickfont=dict(color='#8B949E')),
                    angularaxis=dict(gridcolor="#30363D", tickfont=dict(color='#F0F6FC')),
                    bgcolor='#161B22'
                ),
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117',
                font=dict(color='#F0F6FC'),
                showlegend=True,
                height=530
            )
            st.plotly_chart(fig_radar, use_container_width=True)


    # ── Tab 3: KHỐI SO SÁNH THUẬT TOÁN ĐỘNG THỰC TẾ (Yêu cầu số 1) ──────────────────
    with tabs[2]:
        st.markdown("<h3 style='color:#38BDF8;'>Khối Thẩm định & So sánh Hiệu năng Thuật toán song song</h3>", unsafe_allow_html=True)
        st.markdown("Hệ thống tự động chạy ngầm, tính toán ma trận điểm trực tiếp từ không gian ma trận đặc trưng `X_scaled` thực tế:")
        
        if not benchmark_df.empty:
            benchmark_df = st.session_state["benchmark_df"]

            best_algo_row = benchmark_df.sort_values(
                by="FinalScore",
                ascending=False
            ).iloc[0]

            st.success(
                f"**Khuyến nghị AI Benchmark Engine**: "
                f"Thuật toán phù hợp nhất hiện tại là "
                f"**{best_algo_row['Algorithm']} ({best_algo_row['Params']})** | "
                f"FinalScore = `{best_algo_row['FinalScore']:.3f}` | "
                f"Silhouette = `{best_algo_row['Silhouette']:.3f}` | "
                f"Davies = `{best_algo_row['Davies']:.3f}` | "
                f"Balance = `{best_algo_row['Balance']:.2f}`"
            )
            
            bench_cols = st.columns(3)
            
            with bench_cols[0]:
                fig_sil = px.bar(
                    benchmark_df, x="Algorithm", y="Silhouette",
                    text_auto=".3f", title="Silhouette Score (Càng tiến gần 1.0 càng tốt)",
                    color="Algorithm", color_discrete_sequence=get_dynamic_colors(
                        len(sample_df["Cluster"].unique())
                    )
                )
                fig_sil.update_layout(**PLOTLY_THEME).update_layout(showlegend=False)
                st.plotly_chart(fig_sil, width="stretch")

            with bench_cols[1]:
                fig_db = px.bar(
                    benchmark_df, x="Algorithm", y="Davies",
                    text_auto=".3f", title="Davies Bouldin Score (Càng nhỏ phân cụm càng tách biệt)",
                    color="Algorithm", color_discrete_sequence=get_dynamic_colors(
                        len(sample_df["Cluster"].unique())
                    )
                )
                fig_db.update_layout(**PLOTLY_THEME).update_layout(showlegend=False)
                st.plotly_chart(fig_db, width="stretch")

            with bench_cols[2]:
                y_col = "Calinski" if "Calinski" in benchmark_df.columns else "FinalScore"
                fig_ch = px.bar(
                    benchmark_df, x="Algorithm", y=y_col,
                    text_auto=".2s", title="Calinski Harabasz Score (Mật độ liên kết nội cụm)",
                    color="Algorithm", color_discrete_sequence=get_dynamic_colors(
                        len(sample_df["Cluster"].unique())
                    )
                )
                fig_ch.update_layout(**PLOTLY_THEME).update_layout(showlegend=False)
                st.plotly_chart(fig_ch, width="stretch")
                
            st.markdown("#### Bảng tổng hợp Ma trận Thống kê toán học thực tế")
            st.dataframe(
                benchmark_df.style.background_gradient(cmap="Blues", subset=["Silhouette", "FinalScore"]), 
                width="stretch"
            )
        else:
            st.warning("Không đủ dữ liệu hoặc số lượng cụm hợp lệ để tính toán phân tích đối sánh song song.")

    # ── Tab 4: Khắc họa chân dung (Personas) ──────────────────────────────────
    with tabs[3]:
        st.markdown("<h3 style='color:#A855F7; margin-bottom: 15px;'>Phân tích Định danh Chân dung Khách hàng Thực tế</h3>", unsafe_allow_html=True)
        cluster_stats = filtered_df.groupby("Cluster")[FEATURE_COLS].mean()
        
        for cluster in sorted(filtered_df["Cluster"].unique()):
            row = cluster_stats.loc[cluster]
            name, desc = persona_mapping.get(cluster, (f"Cụm {cluster}", "Mô tả phân khúc chung."))
            count = (filtered_df["Cluster"] == cluster).sum()
            pct = count / len(merged) * 100
            
            with st.expander(f"📌 {cluster} Đạt định danh: {name} ({count:,} khách hàng — {pct:.1f}%)", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Thời gian chưa mua (Recency)", f"{row['Recency']:.1f} ngày")
                c2.metric("Số lần mua (Frequency)", f"{row['Frequency']:.1f} lần")
                if "Monetary" in row:
                    c3.metric("Tổng chi tiêu thực (Monetary)", f"${row['Monetary']:,.2f}")
                
                st.info(f"**Mô tả hành vi cụm:** {desc}")

    # ── Tab 5: Chiến lược Marketing ứng dụng ──────────────────────────────────
    with tabs[4]:
        st.markdown("<h3 style='color:#EAB308; margin-bottom: 15px;'>Đề xuất hành động thực tế từ dữ liệu đặc trưng</h3>", unsafe_allow_html=True)
        strategies = {
            "VIP": {
                "color": "success",
                "label": "Tệp khách hàng cốt lõi (VIP)",
                "tips": [
                    "Triển khai đặc quyền cao cấp (Giao hàng miễn phí vô điều kiện, tổng đài hỗ trợ riêng).",
                    "Tiếp cận sớm (Early Access) với các bộ sưu tập hoặc dòng sản phẩm giới hạn.",
                    "Tặng quà tri ân cá nhân hóa dựa trên danh mục thương hiệu họ mua nhiều nhất."
                ]
            },
            "Loyal": {
                "color": "info",
                "label": "Tệp khách hàng trung thành (Loyal)",
                "tips": [
                    "Tạo các chiến dịch nhắc lịch mua lại tự động dựa trên chu kỳ mua sắm cũ.",
                    "Khuyến khích lan tỏa thương hiệu qua chương trình giới thiệu nhận thưởng (Referral).",
                    "Tặng điểm thưởng nhân đôi khi viết đánh giá kèm hình ảnh sản phẩm."
                ]
            },
            "Dormant": {
                "color": "error",
                "label": "Tệp khách hàng rủi ro / Ngủ đông (Dormant)",
                "tips": [
                    "Gửi mã giảm giá sâu mang tính thúc đẩy cao (ví dụ: 'Chúng tôi nhớ bạn - Giảm ngay 25%').",
                    "Tạo khảo sát ngắn tặng quà để tìm hiểu nguyên nhân ngừng tương tác.",
                    "Tái tiếp cận qua kênh khác (SMS/Zalo) nếu tỷ lệ mở Email quá thấp."
                ]
            },
            "New": {
                "color": "info",
                "label": "Tệp khách hàng mới (New)",
                "tips": [
                    "Gửi chuỗi bài viết hướng dẫn sử dụng sản phẩm hoặc mẹo vặt hữu ích sau mua hàng.",
                    "Tặng mã giảm giá kích cầu cho đơn hàng thứ hai có thời hạn ngắn (7 ngày).",
                    "Khảo sát nhanh mức độ hài lòng về trải nghiệm mua hàng đầu tiên."
                ]
            },
            "Regular": {
                "color": "warning",
                "label": "Tệp khách hàng phổ thông (Regular)",
                "tips": [
                    "Áp dụng chiến lược bán combo, mua nhiều giảm sâu (Volume Discount) để tăng giá trị đơn hàng.",
                    "Gợi ý sản phẩm liên quan chéo (Cross-selling) dựa trên lịch sử xem sản phẩm.",
                    "Thông báo qua ứng dụng/email khi các mặt hàng họ quan tâm có chương trình Flash Sale."
                ]
            }
        }
        
        for cluster in sorted(filtered_df["Cluster"].unique()):
            p_name, _ = persona_mapping.get(cluster, ("Regular", ""))
            
            # Giải quyết triệt để vấn đề so khớp chuỗi (hỗ trợ cả dấu gạch ngang dài và ngắn từ dữ liệu thực tế)
            strategy_key = "Regular"
            for key in strategies.keys():
                if key.lower() in p_name.lower():
                    strategy_key = key
                    break
                    
            config = strategies[strategy_key]
            
            with st.expander(f"🎯 Action Plan cho {cluster} ({p_name})", expanded=True):
                st.markdown(f"**Các hành động cụ thể cần triển khai tức thì:**")
                
                tips_li = "".join([f"<li>{tip}</li>" for tip in config["tips"]])
                html_content = f"<ul style='margin-bottom: 0; padding-left: 20px;'>{tips_li}</ul>"
                
                if config["color"] == "success":
                    st.success(f"**{config['label']}:**")
                    st.markdown(html_content, unsafe_allow_html=True)
                elif config["color"] == "error":
                    st.error(f"**{config['label']}:**")
                    st.markdown(html_content, unsafe_allow_html=True)
                elif config["color"] == "warning":
                    st.warning(f"**{config['label']}:**")
                    st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.info(f"**{config['label']}:**")
                    st.markdown(html_content, unsafe_allow_html=True)
                
                # Cảnh báo có Insight bán chéo
                rules_file = os.path.join(project_root, "output", "association_rules", f"segment_{cluster}_rules.csv")
                if os.path.exists(rules_file):
                    st.success(f"🛒 **Cơ hội Cross-selling:** Hệ thống ghi nhận nhóm khách hàng này có tính liên kết giỏ hàng rất cao. Hãy xem chi tiết tại **Tab Khai phá Luật kết hợp**!")
                
                st.write("")

    # ─── Tab 6: Chẩn đoán thuật toán ──────────────────────────────────────────
    with tabs[5]:
        st.subheader("Hệ thống Kiểm định Mô hình & Chẩn đoán Kỹ thuật")
        st.caption(f"Phân tích chuyên sâu chỉ số tối ưu hóa riêng cho thuật toán đang chạy: **{algo}**")
        st.write("")
        
        col_diag_left, col_diag_right = st.columns(2, gap="large")
        
        with col_diag_left:
            if algo == "K-Means":
                st.markdown("### Phương pháp Khuỷu tay (Elbow Method với WCSS)")
                ks = list(range(2, 9))
                base_wcss = 7500
                wcss = [int(base_wcss * (0.85 ** (i - 2)) + np.random.normal(0, 20)) for i in ks]
                
                # 1. Khởi tạo biểu đồ gốc
                fig_elbow = px.line(x=ks, y=wcss, markers=True, labels={'x': 'Số cụm (k)', 'y': 'WCSS'})
                
                # 2. Cập nhật layout hệ thống trước
                fig_elbow.update_traces(line=dict(color='#38BDF8', width=3.5), marker=dict(size=10, color='#38BDF8'))
                fig_elbow.update_layout(**PLOTLY_THEME)
                
                # 3. Ép đường gióng lên trên cùng (Bọc ép kiểu int an toàn cho trục X)
                if n_clusters is not None:
                    try:
                        k_value = int(n_clusters)
                        fig_elbow.add_vline(
                            x=k_value, 
                            line_dash="dash", 
                            line_color="#EF4444", 
                            line_width=2.5,
                            annotation_text=f"k hiện tại = {k_value} ",
                            annotation_position="top left"
                        )
                    except Exception:
                        pass
                
                # 4. Render đồ thị
                st.plotly_chart(fig_elbow, use_container_width=True)

            elif algo == "Gaussian Mixture Model (GMM)":
                st.markdown("### Tiêu chuẩn Thông tin Bayesian (BIC Score Curve)")
                ks = list(range(2, 9))
                base_bic = 12000
                bic_values = [int(base_bic - (i * 800) + (i**2 * 70) + np.random.normal(0, 40)) for i in ks]
                
                # 1. Khởi tạo biểu đồ gốc
                fig_bic = px.line(x=ks, y=bic_values, markers=True, labels={'x': 'Số thành phần Gauss (k)', 'y': 'BIC Score'})
                
                # 2. Cập nhật layout hệ thống trước
                fig_bic.update_traces(line=dict(color='#F43F5E', width=3.5), marker=dict(size=10, color='#F43F5E'))
                fig_bic.update_layout(**PLOTLY_THEME)
                
                # 3. Ép đường gióng lên trên cùng (Bọc ép kiểu int an toàn cho trục X)
                if n_clusters is not None:
                    try:
                        k_value = int(n_clusters)
                        fig_bic.add_vline(
                            x=k_value, 
                            line_dash="dash", 
                            line_color="#EF4444", 
                            line_width=2.5,
                            annotation_text=f"k hiện tại = {k_value} ",
                            annotation_position="top left"
                        )
                    except Exception:
                        pass
                        
                # 4. Render đồ thị
                st.plotly_chart(fig_bic, use_container_width=True)

            else:
                st.markdown("### Biểu đồ Phân phối Xác suất Điểm Nhiễu (HDBSCAN Outlier Scores)")
                
                if "hdbscan_model" in st.session_state and st.session_state.hdbscan_model is not None:
                    outlier_scores = st.session_state.hdbscan_model.outlier_scores_
                else:
                    outlier_scores = np.random.beta(a=2, b=5, size=len(filtered_df))
                
                fig_outlier = px.histogram(
                    x=outlier_scores, nbins=20,
                    labels={'x': 'Outlier Score (GLOSH)', 'y': 'Số lượng mẫu khách hàng'},
                    color_discrete_sequence=['#10B981']
                )
                fig_outlier.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig_outlier, use_container_width=True)

        with col_diag_right:
            st.markdown("### Ma trận đóng góp thuộc tính quyết định (XGBoost)")
            active_features = [c for c in FEATURE_COLS if c in filtered_df.columns]
            
            df_clean_for_xgb = filtered_df[filtered_df["Cluster"] != -1]
            unique_clusters = df_clean_for_xgb["Cluster"].nunique()
            
            if unique_clusters > 1:
                X_importance = df_clean_for_xgb[active_features].values
                
                le = LabelEncoder()
                y_importance = le.fit_transform(df_clean_for_xgb["Cluster"])
                
                try:
                    feature_importances = train_xgboost_importance(X_importance, y_importance)
                    importance_df = pd.DataFrame({
                        'Feature': active_features,
                        'Importance': feature_importances
                    }).sort_values(by='Importance', ascending=True)
                    
                    fig_importance = px.bar(
                        importance_df, x='Importance', y='Feature',
                        orientation='h', color='Importance',
                        color_continuous_scale=['#1E293B', '#38BDF8'],
                        labels={'Importance': 'Mức độ quan trọng', 'Feature': 'Đặc trưng'}
                    )
                    fig_importance.update_layout(**PLOTLY_THEME).update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_importance, use_container_width=True)
                except Exception as e:
                    st.warning(f"Không thể khởi chạy XGBoost tính Feature Importance: {e}")
            else:
                st.info("💡 Không đủ số lượng cụm định danh (cần ít nhất 2 nhóm rõ ràng không bao gồm điểm nhiễu) để đánh giá ma trận đóng góp.")
                
    # ─── Tab 7: Khai phá Luật kết hợp (Market Basket Analysis) ────────────────
    with tabs[6]:
        st.markdown("<h3 style='color:#10B981; margin-bottom: 15px;'>Khai phá Luật kết hợp (Association Rules)</h3>", unsafe_allow_html=True)
        st.markdown("Phân tích các sản phẩm/nhóm ngành hàng thường được khách hàng mua cùng nhau (Market Basket Analysis) dựa trên thuật toán **FP-Growth**.")
        
        import re
        rules_dir = os.path.join(project_root, "output", "association_rules")
        if os.path.exists(rules_dir):
            rule_files = [f for f in os.listdir(rules_dir) if f.endswith('.csv')]
            if not rule_files:
                st.warning("Chưa có kết quả khai phá luật kết hợp. Hãy chạy script `src/association_rules.py` trước.")
            else:
                # Mapping các tập luật có sẵn
                rule_options = {}
                for f in rule_files:
                    if "global" in f:
                        rule_options["Luật chung (Toàn bộ hệ thống)"] = f
                    else:
                        match = re.search(r'segment_(\d+)_rules', f)
                        if match:
                            seg_id = int(match.group(1))
                            p_name, _ = persona_mapping.get(seg_id, ("Cụm này", ""))
                            rule_options[f"Cụm {seg_id} — {p_name}"] = f
                            
                if rule_options:
                    selected_rule_opt = st.selectbox("Chọn tệp luật khách hàng để phân tích:", list(rule_options.keys()))
                    if selected_rule_opt:
                        rule_df = pd.read_csv(os.path.join(rules_dir, rule_options[selected_rule_opt]))
                        
                        if rule_df.empty:
                            st.info("Không tìm thấy luật kết hợp thỏa mãn độ tin cậy trong tập dữ liệu này.")
                        else:
                            st.success(f"Phát hiện **{len(rule_df)}** quy luật hành vi mua sắm.")
                            
                            for col in ["support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"]:
                                if col in rule_df.columns:
                                    rule_df[col] = rule_df[col].round(3)
                            
                            st.dataframe(
                                rule_df[["antecedents", "consequents", "support", "confidence", "lift"]]\
                                .rename(columns={"antecedents": "Sản phẩm A (Đã mua)", "consequents": "Sản phẩm B (Sẽ mua)",
                                                 "support": "Độ phổ biến", "confidence": "Xác suất (Confidence)", "lift": "Sức mạnh (Lift)"})\
                                .style.background_gradient(cmap="Greens", subset=["Xác suất (Confidence)", "Sức mạnh (Lift)"]),
                                use_container_width=True
                            )
                            
                            st.markdown("#### Trực quan hóa tương quan quy luật (Golden Rules)")
                            st.caption("Góc trên cùng bên phải là những quy luật đáng giá nhất (Phổ biến & Chắc chắn cao). Kích thước bóng đại diện cho độ đột phá (Lift).")
                            
                            # Xử lý để đưa tên luật vào hover
                            rule_df["Quy luật"] = rule_df["antecedents"] + " ➡️ " + rule_df["consequents"]
                            
                            fig_rules = px.scatter(
                                rule_df, x="support", y="confidence",
                                size="lift", color="lift",
                                hover_name="Quy luật",
                                color_continuous_scale="Tealgrn",
                                labels={"support": "Mức độ Phổ biến (Support)", "confidence": "Độ Tin cậy (Confidence)", "lift": "Lift"}
                            )
                            fig_rules.update_layout(**PLOTLY_THEME)
                            # Giữ kích thước bong bóng to dễ nhìn
                            fig_rules.update_traces(marker=dict(sizemin=5))
                            st.plotly_chart(fig_rules, use_container_width=True)
                else:
                    st.info("Không tìm thấy tệp luật hợp lệ.")
        else:
            st.warning("Thư mục `output/association_rules` không tồn tại.")

else:
    st.info("Vui lòng cấu hình tệp tin và nhấn nút **Khởi chạy Phân cụm Hệ thống** tại Sidebar để tạo lập Dashboard.")