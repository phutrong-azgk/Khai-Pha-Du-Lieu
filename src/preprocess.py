import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# SỬA: Đổi category sang object để tránh lỗi Categorical khi xử lý chunk
CONFIG = {
    "dtypes": {
        "user_id": "int64",
        "product_id": "int32",
        "price": "float32",
        "event_type": "object", 
        "brand": "object",
        "category_code": "object"
    },
    "required_cols": ["user_id", "product_id", "price", "event_time", "event_type"],
    "dedup_subset": ["user_session", "product_id", "event_time"],
    "winsor_price_threshold": 10000.0,
    "output_meta": "output/cleaning_report.json"
}

def load_and_clean(filepath: str, output_path: str = "output/clean_purchase_data.csv") -> pd.DataFrame:
    logger.info("Khởi động tiến trình làm sạch dữ liệu...")
    
    chunk_list = []
    report = {"raw_rows": 0, "purchase_rows": 0, "duplicates_removed": 0, "invalid_prices_removed": 0, "null_rows_removed": 0}

    # Đọc theo chunk để tránh tràn RAM
    for chunk in pd.read_csv(filepath, chunksize=500000, dtype=CONFIG["dtypes"]):
        report["raw_rows"] += len(chunk)
        
        # Chỉ giữ lại các giao dịch mua hàng
        chunk = chunk[chunk["event_type"] == "purchase"].copy()
        report["purchase_rows"] += len(chunk)

        # Xử lý thời gian
        before = len(chunk)
        chunk["event_time"] = pd.to_datetime(chunk["event_time"], utc=True, errors="coerce")
        chunk.dropna(subset=["event_time"], inplace=True)
        report["null_rows_removed"] += (before - len(chunk))

        chunk_list.append(chunk)

    df = pd.concat(chunk_list, ignore_index=True)

    # Lọc giá trị giá bất khả thi
    before_price = len(df)
    df = df[(df["price"] > 0) & (df["price"] < CONFIG["winsor_price_threshold"])]
    report["invalid_prices_removed"] = (before_price - len(df))

    # Xử lý null cột brand/category bằng cách gán giá trị mặc định
    df[["brand", "category_code"]] = df[["brand", "category_code"]].fillna("unknown")

    # Deduplicate
    before_dedup = len(df)
    df.drop_duplicates(subset=CONFIG["dedup_subset"], inplace=True)
    report["duplicates_removed"] = (before_dedup - len(df))

    # Tối ưu hóa bộ nhớ sau khi làm sạch xong
    for col in ["event_type", "brand", "category_code"]:
        df[col] = df[col].astype("category")

    # Xuất báo cáo
    os.makedirs(os.path.dirname(CONFIG["output_meta"]), exist_ok=True)
    with open(CONFIG["output_meta"], "w") as f:
        json.dump(report, f, indent=4)

    # Lưu output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Hoàn thành. Báo cáo lưu tại {CONFIG['output_meta']}")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        # Đảm bảo file events.csv nằm đúng thư mục data
        load_and_clean("data/events.csv")
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")