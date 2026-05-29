import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

# Cấu hình logger để sử dụng trong các module khác
logger = logging.getLogger(__name__)

# Cấu hình tập trung thông qua Object (Config-driven architecture)
PIPELINE_CONFIG = {
    "features": {
        "required_schema": ["user_id", "event_time", "product_id", "price", "brand", "category_code"],
        "target_cols": [
            "Recency", "Frequency", "Monetary", "Average_Basket_Value", "Brand_Diversity", "Category_Diversity"
        ]
    },
    "preprocessing": {
        "winsor_lower": 0.01,
        "winsor_upper": 0.99,
        "skew_threshold": 0.75,
        "pca_variance_target": 4
    }
}

def build_features(df: pd.DataFrame, output_path: str = "output/customer_features.csv") -> pd.DataFrame:
    """
    Xây dựng bảng đặc trưng theo từng user_id từ tập dữ liệu thô.
    """
    logger.info("Bắt đầu tiến trình trích xuất đặc trưng khách hàng...")
    
    # Schema Validation
    for col in PIPELINE_CONFIG["features"]["required_schema"]:
        if col not in df.columns:
            raise KeyError(f"Schema Validation Error: Thiếu cột bắt buộc '{col}' trong dữ liệu đầu vào.")

    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    snapshot_date = df["event_time"].max()

    # --- RFM cơ bản ---
    rfm = df.groupby("user_id").agg(
        Recency   = ("event_time", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("product_id", "count"),
        Monetary  = ("price", "sum"),
    ).reset_index()

    # --- Đặc trưng mở rộng ---
    logger.info("Tính toán đặc trưng Interpurchase_Time bằng Vectorized Operation...")
    df_sorted = df.sort_values(by=["user_id", "event_time"])
    
    df_sorted["gap_days"] = df_sorted.groupby("user_id")["event_time"].diff().dt.total_seconds() / 86400.0
    ipt = df_sorted.groupby("user_id")["gap_days"].mean().reset_index()
    ipt.columns = ["user_id", "Interpurchase_Time"]

    brand_div = df.groupby("user_id")["brand"].nunique().reset_index()
    brand_div.columns = ["user_id", "Brand_Diversity"]

    cat_div = df.groupby("user_id")["category_code"].nunique().reset_index()
    cat_div.columns = ["user_id", "Category_Diversity"]

    abv = rfm[["user_id", "Monetary", "Frequency"]].copy()
    abv["Average_Basket_Value"] = abv["Monetary"] / abv["Frequency"]
    abv = abv[["user_id", "Average_Basket_Value"]]

    # --- Gộp tập dữ liệu ---
    features = rfm.copy()
    for extra in [ipt, brand_div, cat_div, abv]:
        features = features.merge(extra, on="user_id", how="left")

    features.fillna(0, inplace=True)

    if (features["Frequency"] < 0).any() or (features["Monetary"] < 0).any():
        raise ValueError("Sanity Check Failed: Phát hiện giá trị âm tại trường dữ liệu Frequency hoặc Monetary.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.info(f"Lưu đặc trưng thành công: {output_path} [{len(features):,} khách hàng]")
    return features


def scale_and_reduce(
    features: pd.DataFrame,
    scaler_path: str = "models/scaler.pkl",
    pca_path: str = "models/pca.pkl",
    pca_meta_path: str = "output/pca_metadata.json",
    output_pca_path: str = "output/pca_data.csv",
    output_scaled_path: str = "output/scaled_features.csv"
):
    """
    RobustScaler + Adaptive PCA, lưu trữ toàn bộ mô hình và dữ liệu đã chuẩn hóa.
    """
    logger.info("Chuẩn hoá và xử lý phân phối dữ liệu...")

    feature_cfg = PIPELINE_CONFIG["features"]
    prep_cfg = PIPELINE_CONFIG["preprocessing"]
    
    feature_cols = feature_cfg["target_cols"].copy()
    X_df = features[feature_cols].copy()

    # Winsorization
    low_q = prep_cfg["winsor_lower"]
    high_q = prep_cfg["winsor_upper"]
    
    # 1. Winsorization: Áp dụng cho toàn bộ đặc trưng
    cols_to_winsorize = feature_cols
    for col in cols_to_winsorize:
        if col in X_df.columns:
            q_low = X_df[col].quantile(low_q)
            q_high = X_df[col].quantile(high_q)
            X_df[col] = X_df[col].clip(q_low, q_high)
            logger.info(f"    -> Winsorized feature: {col}")

    # 2. Log Transform: Áp dụng cho toàn bộ đặc trưng có độ lệch (skew) lớn
    cols_to_log = feature_cols
    for col in cols_to_log:
        if col in X_df.columns:
            # Kiểm tra độ lệch (skew) để quyết định có áp dụng log hay không
            if X_df[col].skew() > prep_cfg["skew_threshold"]:
                X_df[col] = np.log1p(X_df[col])
                logger.info(f"    -> Applied Log1p on skewed feature: {col}")

    if not np.isfinite(X_df.values).all():
        raise ValueError("ML Preprocessing Error: Dữ liệu chứa các ký tự không hợp lệ như NaN hoặc Infinite (inf).")

    # RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    scaler.feature_names_in_ = np.array(feature_cols)
    scaler.n_features_in_ = len(feature_cols)

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    scaled_df.insert(0, "user_id", features["user_id"].values)
    scaled_df.to_csv(output_scaled_path, index=False)
    logger.info(f"Đã kết xuất tập dữ liệu chuẩn hóa xuống đĩa: {output_scaled_path}")

    # Adaptive PCA
    pca = PCA(n_components=prep_cfg["pca_variance_target"], random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    n_components_kept = pca.n_components_
    explained_variance = pca.explained_variance_ratio_.tolist()
    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_).tolist()

    feature_importance = {}
    for i in range(n_components_kept):
        raw_loadings = pca.components_[i]
        abs_weights = np.abs(raw_loadings)
        normalized_weights = abs_weights / abs_weights.sum()
        feature_importance[f"PC{i+1}"] = dict(zip(feature_cols, normalized_weights.tolist()))

    pca_metadata = {
        "n_components": int(n_components_kept),
        "feature_importance_mapping": feature_importance,
        "explained_variance_ratio": explained_variance,
        "cumulative_explained_variance_ratio": cum_explained_variance
    }
    
    os.makedirs(os.path.dirname(pca_meta_path), exist_ok=True)
    with open(pca_meta_path, "w", encoding="utf-8") as f:
        json.dump(pca_metadata, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    logger.info(f"    Lưu artifacts thành công: {scaler_path} & {pca_path}")

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components_kept)])
    pca_df.insert(0, "user_id", features["user_id"].values)
    pca_df.to_csv(output_pca_path, index=False)
    logger.info(f"Lưu không gian PCA thành công: {output_pca_path}")

    return X_scaled, X_pca, pca_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    input_data_path = "output/clean_purchase_data.csv"
    
    if os.path.exists(input_data_path):
        df = pd.read_csv(input_data_path)
        features = build_features(df)
        X_scaled, X_pca, pca_df = scale_and_reduce(features)
        logger.info("Toàn bộ luồng Feature Engineering đã thực thi an toàn đạt chuẩn Production.")
    else:
        logger.error(f"Không tìm thấy tệp đầu vào tại: {input_data_path}.")