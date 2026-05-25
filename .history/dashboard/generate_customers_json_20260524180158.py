import pandas as pd
import json
import os
import numpy as np
import joblib
import datetime

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# ==========================================================
# HDBSCAN (optional)
# ==========================================================
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("Chưa cài hdbscan. Cài bằng: pip install hdbscan")

print("=" * 70)
print("ĐANG TẠO CUSTOMERS.JSON CHO DASHBOARD")
print("Đồng bộ hoàn toàn với pipeline Python và Streamlit")
print("=" * 70)

# ==========================================================
# PATHS
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

os.makedirs(os.path.join(SCRIPT_DIR, "data"), exist_ok=True)

features_path = os.path.join(PROJECT_ROOT, "output", "customer_features.csv")
models_dir = os.path.join(PROJECT_ROOT, "models")
output_path = os.path.join(SCRIPT_DIR, "data", "customers.json")

print(f"📂 Features file : {features_path}")
print(f"📂 Models dir    : {models_dir}")
print(f"📂 Output file   : {output_path}")

# ==========================================================
# CHECK INPUT FILE
# ==========================================================
if not os.path.exists(features_path):
    print(f"\n Không tìm thấy file:\n   {features_path}")
    print("Hãy chạy pipeline trước:")
    print("   python src/preprocess.py")
    print("   python src/feature_engineering.py")
    print("   python src/clustering.py")
    print("   python src/evaluation.py")
    exit()

# ==========================================================
# LOAD FEATURES
# ==========================================================
df = pd.read_csv(features_path)
print(f"Đã load {len(df):,} khách hàng")

# ==========================================================
# FEATURE COLUMNS
# PHẢI GIỐNG HỆT feature_engineering.py
# ==========================================================
FEATURE_COLS = [
    "Recency", 
    "Frequency", 
    "Monetary", 
    "Average_Basket_Value"
]

# Chỉ giữ các cột thực sự tồn tại
feature_cols = [col for col in FEATURE_COLS if col in df.columns]

print(f"Sử dụng {len(feature_cols)} features:")
print(f"   {feature_cols}")

if len(feature_cols) == 0:
    print(" Không tìm thấy feature columns.")
    exit()

# ==========================================================
# CHUẨN BỊ DỮ LIỆU
# Không tự log1p ở đây vì feature_engineering.py đã lưu scaler.pkl
# với toàn bộ pipeline phù hợp. Chỉ cần đưa dữ liệu gốc vào scaler.
# ==========================================================
X = df[feature_cols].values
print("Sử dụng dữ liệu gốc và để scaler xử lý theo pipeline đã huấn luyện")

# ==========================================================
# LOAD MODELS
# ==========================================================
scaler = None
pca = None
kmeans_model = None

try:
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    pca = joblib.load(os.path.join(models_dir, "pca.pkl"))

    kmeans_path = os.path.join(models_dir, "kmeans.pkl")
    if os.path.exists(kmeans_path):
        kmeans_model = joblib.load(kmeans_path)

    print("Đã load models thành công")
except Exception as e:
    print(f"Không load được model: {e}")

# ==========================================================
# SCALE + PCA
# ==========================================================
# ==========================================================
# PCA
# ==========================================================
if scaler is not None and pca is not None:
    try:
        # -----------------------------------------------
        # 1. Biến đổi dữ liệu theo đúng pipeline đã huấn luyện
        # -----------------------------------------------
        X_scaled_for_clustering = scaler.transform(X)

        # -----------------------------------------------
        # 2. PCA model cũ có thể được train trên 7 features,
        #    trong khi scaler hiện tại chỉ dùng 4 features.
        #    Nếu PCA lỗi, tạo PCA mới trực tiếp từ dữ liệu
        #    đã được scaler transform.
        # -----------------------------------------------
        try:
            X_pca = pca.transform(X_scaled_for_clustering)
            print("Đã dùng PCA từ model")
        except Exception as e:
            print(f"PCA model không tương thích: {e}")
            print("Tạo PCA mới từ dữ liệu hiện tại...")

            from sklearn.decomposition import PCA

            pca_temp = PCA(n_components=2, random_state=42)
            X_pca = pca_temp.fit_transform(X_scaled_for_clustering)

            print("Đã tạo PCA mới thành công")

        # -----------------------------------------------
        # 3. Lưu tọa độ để HTML và Streamlit hiển thị giống nhau
        # -----------------------------------------------
        df["x"] = X_pca[:, 0]
        df["y"] = X_pca[:, 1]

        # -----------------------------------------------
        # 4. Dữ liệu dùng cho clustering
        # -----------------------------------------------
        X_scaled = X_scaled_for_clustering

    except Exception as e:
        print(f"Lỗi transform scaler/pca: {e}")

        # Nếu scaler lỗi hoàn toàn thì dùng dữ liệu gốc
        X_scaled = X

        # Tạo PCA mới từ dữ liệu gốc
        from sklearn.decomposition import PCA

        pca_temp = PCA(n_components=2, random_state=42)
        X_pca = pca_temp.fit_transform(X_scaled)

        df["x"] = X_pca[:, 0]
        df["y"] = X_pca[:, 1]

        print("Đã tạo PCA mới từ dữ liệu gốc")

else:
    print("Không có scaler/pca → Dùng dữ liệu gốc")

    X_scaled = X

    from sklearn.decomposition import PCA

    pca_temp = PCA(n_components=2, random_state=42)
    X_pca = pca_temp.fit_transform(X_scaled)

    df["x"] = X_pca[:, 0]
    df["y"] = X_pca[:, 1]

    print("Đã tạo PCA mới")

# ==========================================================
# CLUSTERING
# ==========================================================
print("\nĐang chạy clustering...")

scores = {
    "kmeans": {},
    "gmm": {},
    "hdbscan": None
}

best_algorithm = "K-Means"
best_k = 2
best_score = -1.0

# ==========================================================
# 1. K-MEANS (K = 2 → 9)
# ==========================================================
print("\nK-MEANS")

for k in range(2, 10):
    print(f"   → K = {k}")

    # Kiểm tra model KMeans đã lưu có thực sự tương thích hay không
    use_saved_model = (
        kmeans_model is not None
        and hasattr(kmeans_model, "n_clusters")
        and hasattr(kmeans_model, "n_features_in_")
        and k == kmeans_model.n_clusters
        and X_scaled.shape[1] == kmeans_model.n_features_in_
    )

    if use_saved_model:
        try:
            print("Dùng model KMeans đã lưu")
            labels = kmeans_model.predict(X_scaled)
        except Exception as e:
            print(f"Model đã lưu không tương thích: {e}")
            print("Huấn luyện lại KMeans tạm thời...")

            km = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )
            labels = km.fit_predict(X_scaled)
    else:
        # Nếu model không phù hợp thì huấn luyện mới
        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = km.fit_predict(X_scaled)

    df[f"k{k}"] = labels

    try:
        score = silhouette_score(X_scaled, labels)
        score = round(float(score), 4)
    except Exception:
        score = 0.0

    scores["kmeans"][str(k)] = score
    print(f"      Silhouette = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k
        best_algorithm = "K-Means"

# ==========================================================
# 2. GMM (K = 2 → 9)
# ==========================================================
print("\nGAUSSIAN MIXTURE MODEL")

for k in range(2, 10):
    print(f"   → K = {k}")

    try:
        gmm = GaussianMixture(
            n_components=k,
            random_state=42
        )

        labels = gmm.fit_predict(X_scaled)
        df[f"g{k}"] = labels

        score = silhouette_score(X_scaled, labels)
        score = round(float(score), 4)

    except Exception as e:
        print(f"      Lỗi: {e}")
        df[f"g{k}"] = np.zeros(len(df), dtype=int)
        score = 0.0

    scores["gmm"][str(k)] = score
    print(f"      Silhouette = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k
        best_algorithm = "GMM"

# ==========================================================
# 3. HDBSCAN
# ==========================================================
print("\nHDBSCAN")

if HAS_HDBSCAN:
    try:
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10
        )

        labels = hdb.fit_predict(X_scaled)
        df["hdb"] = labels

        non_noise = set(labels) - {-1}

        if len(non_noise) >= 2:
            mask = labels != -1
            score = silhouette_score(
                X_scaled[mask],
                labels[mask]
            )
            score = round(float(score), 4)
        else:
            score = 0.0

        scores["hdbscan"] = score
        print(f"      Silhouette = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = len(non_noise)
            best_algorithm = "HDBSCAN"

    except Exception as e:
        print(f"      Lỗi HDBSCAN: {e}")
        df["hdb"] = np.zeros(len(df), dtype=int)
        scores["hdbscan"] = 0.0
else:
    print("      Bỏ qua (chưa cài hdbscan)")
    df["hdb"] = np.zeros(len(df), dtype=int)
    scores["hdbscan"] = 0.0

# ==========================================================
# BEST RESULT
# ==========================================================
print(f"\nThuật toán tốt nhất : {best_algorithm}")
print(f"K tối ưu            : {best_k}")
print(f"Silhouette Score    : {best_score:.4f}")

# ==========================================================
# EXPORT DATA
# ==========================================================
export_data = []

for _, row in df.iterrows():
    row_data = {
        "user_id": int(row["user_id"]) if "user_id" in row else 0,

        # RFM
        "R": int(row.get("Recency", 0)),
        "F": int(row.get("Frequency", 0)),
        "M": round(float(row.get("Monetary", 0)), 2),

        # Extra features
        "Interpurchase_Time": round(float(row.get("Interpurchase_Time", 0)), 4),
        "Brand_Diversity": int(row.get("Brand_Diversity", 0)),
        "Category_Diversity": int(row.get("Category_Diversity", 0)),
        "Average_Basket_Value": round(
            float(row.get("Average_Basket_Value", 0)), 2
        ),

        # PCA coordinates
        "x": round(float(row.get("x", 0)), 4),
        "y": round(float(row.get("y", 0)), 4),

        # HDBSCAN
        "hdb": int(row.get("hdb", 0)),
    }

    # K-Means labels
    for k in range(2, 10):
        row_data[f"k{k}"] = int(row.get(f"k{k}", 0))

    # GMM labels
    for k in range(2, 10):
        row_data[f"g{k}"] = int(row.get(f"g{k}", 0))

    export_data.append(row_data)

# ==========================================================
# OUTPUT JSON
# ==========================================================
output = {
    "metadata": {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_features": features_path,
        "models_dir": models_dir,
        "total_customers": len(export_data),
        "feature_columns": feature_cols,
        "best_algorithm": best_algorithm,
        "best_k": int(best_k),
        "best_silhouette_score": float(best_score),
    },

    "data": export_data,
    "scores": scores,

    "best_algorithm": best_algorithm,
    "best_k": int(best_k),
    "silhouette_score": float(best_score),

    "total_customers": len(export_data),
}

# ==========================================================
# SAVE JSON
# ==========================================================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# ==========================================================
# DONE
# ==========================================================
print("\n" + "=" * 70)
print("HOÀN THÀNH TẠO CUSTOMERS.JSON THÀNH CÔNG")
print("=" * 70)
print(f"File JSON: {output_path}")
print(f"Tổng khách hàng: {len(export_data):,}")
print(f"Thuật toán tốt nhất: {best_algorithm}")
print(f"K tối ưu: {best_k}")
print(f"Silhouette Score: {best_score:.4f}")
print("=" * 70)