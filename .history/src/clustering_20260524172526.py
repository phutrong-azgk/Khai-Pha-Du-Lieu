import os
import hdbscan
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

# 🎯 CỐ ĐỊNH BỘ ĐẶC TRƯNG LÕI THEO YÊU CẦU
FIXED_FEATURE_COLS = [
    "Recency", 
    "Frequency", 
    "Monetary", 
    "Interpurchase_Time",
    "Brand_Diversity", 
    "Category_Diversity", 
    "Average_Basket_Value"
]

# ==========================================================
# K-MEANS CHO k = 2..9
# ==========================================================
def run_kmeans_all(X_scaled: np.ndarray):
    results = {}
    models = {}
    scores = {}

    print("\n=== K-MEANS (k = 2..9) ===")
    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        results[f"k{k}"] = labels
        models[f"k{k}"] = model

        # Tính toán điểm Silhouette để tìm cấu hình K tốt nhất
        score = silhouette_score(X_scaled, labels)
        scores[f"k{k}"] = score

        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"K-Means k={k} (Sil={score:.4f}): {distribution}")

    return results, models, scores


# ==========================================================
# GMM CHO k = 2..9
# ==========================================================
def run_gmm_all(X_scaled: np.ndarray):
    results = {}
    models = {}
    scores = {}

    print("\n=== GMM (k = 2..9) ===")
    for k in range(2, 10):
        model = GaussianMixture(
            n_components=k, covariance_type="full", random_state=42
        )
        labels = model.fit_predict(X_scaled)

        results[f"g{k}"] = labels
        models[f"g{k}"] = model

        # Tính toán điểm Silhouette đánh giá cấu hình GMM
        score = silhouette_score(X_scaled, labels)
        scores[f"g{k}"] = score

        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"GMM k={k} (Sil={score:.4f}): {distribution}")

    return results, models, scores


# ==========================================================
# 🔌 WRAPPER FUNCTIONS (CẤP CHO DASHBOARD IMPORT ĐỂ ĐỒNG BỘ)
# ==========================================================
def run_kmeans(X_scaled: np.ndarray, n_clusters: int = 5):
    """Hàm wrapper hỗ trợ giao diện tính toán động thuật toán K-Means"""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return labels, model


def run_gmm(X_scaled: np.ndarray, n_clusters: int = 5):
    """Hàm wrapper hỗ trợ giao diện tính toán động thuật toán GMM"""
    model = GaussianMixture(
        n_components=n_clusters, covariance_type="full", random_state=42
    )
    labels = model.fit_predict(X_scaled)
    return labels, model


def run_hdbscan(X_scaled: np.ndarray, min_cluster_size: int = 50):
    """Hàm wrapper hỗ trợ cả tiến trình chính lẫn giao diện tính toán động HDBSCAN"""
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=10, metric="euclidean"
    )
    labels = model.fit_predict(X_scaled)

    # Đoạn log thống kê nhanh phục vụ khi chạy pipeline nền
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN: {n_clusters} cụm, {n_noise} noise points")
    
    return labels, model


# ==========================================================
# TIẾN TRÌNH HUẤN LUYỆN ĐA THUẬT TOÁN & TỰ ĐỘNG LƯU MÔ HÌNH
# ==========================================================
def run_all_clustering(
    features: pd.DataFrame,
    X_scaled: np.ndarray,
    output_path: str = "output/clustered_customers.csv",
    model_dir: str = "models",
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Khởi tạo bản lưu kết quả ánh xạ với user_id
    result = features[["user_id"]].copy()

    # ------------------------------------------------------
    # Khởi chạy phân cụm với K-Means
    # ------------------------------------------------------
    kmeans_results, kmeans_models, kmeans_scores = run_kmeans_all(X_scaled)
    for col, labels in kmeans_results.items():
        result[col] = labels

    # Tự động tìm mô hình K-Means có Silhouette Score cao nhất để xuất file pkl
    best_k_col = max(kmeans_scores, key=kmeans_scores.get)
    joblib.dump(kmeans_models[best_k_col], f"{model_dir}/kmeans.pkl")
    print(
        f"Tự động lưu mô hình K-Means tốt nhất ({best_k_col}) vào kmeans.pkl"
    )

    # ------------------------------------------------------
    # Khởi chạy phân cụm với GMM
    # ------------------------------------------------------
    gmm_results, gmm_models, gmm_scores = run_gmm_all(X_scaled)
    for col, labels in gmm_results.items():
        result[col] = labels

    # Tự động tìm mô hình GMM có Silhouette Score cao nhất để xuất file pkl
    best_g_col = max(gmm_scores, key=gmm_scores.get)
    joblib.dump(gmm_models[best_g_col], f"{model_dir}/gmm.pkl")
    print(f"Tự động lưu mô hình GMM tốt nhất ({best_g_col}) vào gmm.pkl")

    # ------------------------------------------------------
    # Khởi chạy phân cụm mật độ với HDBSCAN
    # ------------------------------------------------------
    hdb_labels, hdb_model = run_hdbscan(X_scaled)
    result["hdb"] = hdb_labels
    result["cluster_hdbscan"] = hdb_labels  # Alias tương thích ngược
    joblib.dump(hdb_model, f"{model_dir}/hdbscan.pkl")
    print("Đã lưu mô hình HDBSCAN vào hdbscan.pkl")

    # ------------------------------------------------------
    # Lưu bảng tổng hợp nhãn phân cụm ra file CSV
    # ------------------------------------------------------
    result.to_csv(output_path, index=False)
    print(f"\nĐã lưu bảng nhãn phân cụm thành công: {output_path}")
    print(f"Cấu trúc dữ liệu đầu ra (Shape): {result.shape}")
    return result


# ==========================================================
# KHỐI THỰC THI CHÍNH (MAIN)
# ==========================================================
if __name__ == "__main__":
    print("📂 Đang tiến hành đọc tệp đặc trưng khách hàng...")

    # 1. Tải tập dữ liệu đặc trưng hành vi
    features = pd.read_csv("output/customer_features.csv")

    # Kiểm tra kiểm định cột thực tế tồn tại trong file csv đầu vào
    FEATURE_COLS = [
        col for col in FIXED_FEATURE_COLS if col in features.columns
    ]
    print(f"📊 Thuộc tính thực tế sử dụng phân cụm ({len(FEATURE_COLS)}):")
    print(FEATURE_COLS)

    # Tạo DataFrame xử lý độc lập cho không gian thuật toán
    X_df = features[FEATURE_COLS].copy()

    # 2. XỬ LÝ OUTLIERS (Winsorization cắt biên ở mức 1% và 99%)
    cols_to_clip = ["Monetary", "Frequency", "Average_Basket_Value"]
    for col in cols_to_clip:
        if col in X_df.columns:
            q01 = X_df[col].quantile(0.01)
            q99 = X_df[col].quantile(0.99)
            X_df[col] = X_df[col].clip(q01, q99)
            print(f"✂️ {col}: Tiến hành clip từ khoảng [{q01:.2f}, {q99:.2f}]")
    print("✅ Hoàn thành cấu trúc Winsorization dữ liệu nhiễu biên.")

    # 3. ĐỒNG BỘ HÓA LOG TRANSFORMATION
    # Áp dụng log1p cho toàn bộ 4 biến để kéo phẳng phân phối, tối ưu hóa khoảng cách Euclidean
    for col in FEATURE_COLS:
        X_df[col] = np.log1p(X_df[col])
        print(f"📈 {col}: Áp dụng hàm log1p() đồng bộ cấu trúc phân phối")
    print("✅ Hoàn thành Log Transformation đồng bộ hệ thống.")

    # 4. CHUẨN HÓA KHÔNG GIAN DỮ LIỆU VỚI ROBUSTSCALER
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    # Lưu đè file scaler chuẩn để Dashboard đồng bộ không gian transform dữ liệu mới
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Hoàn thành RobustScaler & Cập nhật mô hình tại models/scaler.pkl")

    # 5. KHỞI CHẠY TIẾN TRÌNH PHÂN CỤM TỔNG THỂ
    print("🚀 Bắt đầu khởi chạy phân cụm trên tập dữ liệu đã chuẩn hóa...")
    run_all_clustering(features, X_scaled)

    print("🎉 Tiến trình phân cụm đã hoàn tất thành công và đồng bộ!")