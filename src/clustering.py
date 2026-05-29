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
    "Average_Basket_Value",
    "Brand_Diversity",
    "Category_Diversity"
]

# ==========================================================
# K-MEANS CHO k = 2..9
# ==========================================================
def run_kmeans_all(X_scaled: np.ndarray):
    results = {}
    models = {}

    print("\n=== K-MEANS (k = 2..9) ===")
    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        results[f"k{k}"] = labels
        models[f"k{k}"] = model

        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"K-Means k={k}: {distribution}")

    return results, models, {}


# ==========================================================
# GMM CHO k = 2..9
# ==========================================================
def run_gmm_all(X_scaled: np.ndarray):
    results = {}
    models = {}

    print("\n=== GMM (k = 2..9) ===")
    for k in range(2, 10):
        model = GaussianMixture(
            n_components=k, covariance_type="full", random_state=42
        )
        labels = model.fit_predict(X_scaled)

        results[f"g{k}"] = labels
        models[f"g{k}"] = model

        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"GMM k={k}: {distribution}")

    return results, models, {}


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
    X_pca: np.ndarray,
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
    kmeans_results, kmeans_models, _ = run_kmeans_all(X_scaled)
    for col, labels in kmeans_results.items():
        result[col] = labels

    # ------------------------------------------------------
    # Khởi chạy phân cụm với GMM
    # ------------------------------------------------------
    gmm_results, gmm_models, _ = run_gmm_all(X_scaled)
    for col, labels in gmm_results.items():
        result[col] = labels

    # ------------------------------------------------------
    # Khởi chạy phân cụm mật độ với HDBSCAN
    # ------------------------------------------------------
    hdb_labels, hdb_model = run_hdbscan(X_pca)
    result["hdb"] = hdb_labels
    result["cluster_hdbscan"] = hdb_labels  # Alias tương thích ngược

    # ------------------------------------------------------
    # Đánh giá và lưu mô hình tốt nhất bằng clustering_evaluator
    # ------------------------------------------------------
    from clustering_evaluator import run_clustering_experiments
    eval_df = run_clustering_experiments(X_scaled, X_pca)
    
    if not eval_df.empty:
        best_k_row = eval_df[eval_df["Algorithm"] == "K-Means"].iloc[0]
        best_k_col = f"k{int(best_k_row['K'])}"
        joblib.dump(kmeans_models[best_k_col], f"{model_dir}/kmeans.pkl")
        print(f"Tự động lưu mô hình K-Means tốt nhất ({best_k_col}) vào kmeans.pkl")

        best_g_row = eval_df[eval_df["Algorithm"] == "GMM"].iloc[0]
        best_g_col = f"g{int(best_g_row['K'])}"
        joblib.dump(gmm_models[best_g_col], f"{model_dir}/gmm.pkl")
        print(f"Tự động lưu mô hình GMM tốt nhất ({best_g_col}) vào gmm.pkl")

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

    # 1. Tải tập dữ liệu đặc trưng hành vi và dữ liệu đã chuẩn hóa
    features = pd.read_csv("output/customer_features.csv")
    scaled_df = pd.read_csv("output/scaled_features.csv")
    pca_df = pd.read_csv("output/pca_data.csv")

    X_scaled = scaled_df.drop(columns=["user_id"]).values
    X_pca = pca_df.drop(columns=["user_id"]).values

    print("🚀 Bắt đầu khởi chạy phân cụm trên tập dữ liệu đã chuẩn hóa...")
    run_all_clustering(features, X_scaled, X_pca)

    print("🎉 Tiến trình phân cụm đã hoàn tất thành công và đồng bộ!")