import os
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings

warnings.filterwarnings("ignore")

# Cấu hình
PURCHASE_DATA_PATH = "output/clean_purchase_data.csv"
CLUSTER_DATA_PATH = "output/clustered_customers.csv"
OUTPUT_DIR = "output/association_rules"

TARGET_CLUSTER_COL = "k3"  # K-Means k=3 là mô hình tốt nhất hiện tại
MIN_SUPPORT_GLOBAL = 0.005
MIN_SUPPORT_SEGMENT = 0.01

def load_and_prepare_data():
    print("📂 Đang tải dữ liệu giao dịch...")
    df = pd.read_csv(PURCHASE_DATA_PATH)
    
    # Lọc chỉ các giao dịch mua hàng (mặc dù clean_purchase_data có thể đã lọc)
    df = df[df["event_type"] == "purchase"]
    
    # Loại bỏ các dòng thiếu category_code
    df = df.dropna(subset=["category_code", "user_session", "user_id"])
    
    print(f"✅ Đã tải {len(df)} giao dịch mua hàng hợp lệ.")
    return df

def run_fpgrowth_and_rules(transactions, min_support, name="Global"):
    if len(transactions) == 0:
        print(f"⚠️ {name}: Không có giao dịch nào.")
        return pd.DataFrame()
        
    print(f"🚀 {name}: Chạy FP-Growth trên {len(transactions)} giỏ hàng...")
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_basket = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Chạy FP-Growth
    frequent_itemsets = fpgrowth(df_basket, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        print(f"⚠️ {name}: Không tìm thấy Frequent Itemsets với min_support={min_support}.")
        return pd.DataFrame()
        
    # Tạo Association Rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Sắp xếp theo độ tin cậy và Lift
    if not rules.empty:
        rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False])
        
        # Chuyển đổi frozenset thành chuỗi để lưu CSV
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        
        print(f"🌟 {name}: Tìm thấy {len(rules)} luật kết hợp.")
    else:
        print(f"⚠️ {name}: Không tìm thấy luật kết hợp thỏa mãn điều kiện.")
        
    return rules

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = load_and_prepare_data()
    
    # Tạo danh sách các giỏ hàng (Basket) dựa trên user_session
    print("🛒 Gom nhóm giao dịch theo phiên (Session-based Baskets)...")
    basket_df = df.groupby('user_session')['category_code'].apply(list).reset_index()
    baskets_global = basket_df['category_code'].tolist()
    
    # 1. KHAI PHÁ LUẬT CHUNG (GLOBAL)
    global_rules = run_fpgrowth_and_rules(baskets_global, MIN_SUPPORT_GLOBAL, name="Global")
    if not global_rules.empty:
        global_path = os.path.join(OUTPUT_DIR, "global_rules.csv")
        global_rules.to_csv(global_path, index=False, encoding="utf-8-sig")
        print(f"💾 Đã lưu Global Rules tại: {global_path}")
        
    # 2. KHAI PHÁ LUẬT THEO CỤM KHÁCH HÀNG (SEGMENT-BASED)
    if os.path.exists(CLUSTER_DATA_PATH):
        print("\n👥 Tải nhãn phân cụm khách hàng...")
        clusters = pd.read_csv(CLUSTER_DATA_PATH)
        
        if TARGET_CLUSTER_COL in clusters.columns:
            # Map user_session -> user_id -> cluster
            user_cluster_map = clusters.set_index('user_id')[TARGET_CLUSTER_COL].to_dict()
            
            # Tính map cho từng transaction (lấy user_id)
            session_user_map = df.drop_duplicates('user_session').set_index('user_session')['user_id'].to_dict()
            
            # Gán cluster cho basket_df
            basket_df['user_id'] = basket_df['user_session'].map(session_user_map)
            basket_df['cluster'] = basket_df['user_id'].map(user_cluster_map)
            
            # Xoá các basket không có cluster
            basket_df = basket_df.dropna(subset=['cluster'])
            
            unique_clusters = sorted(basket_df['cluster'].unique())
            for c in unique_clusters:
                segment_baskets = basket_df[basket_df['cluster'] == c]['category_code'].tolist()
                
                # Cụm nhỏ thường cần min_support lớn hơn chút để tránh nhiễu
                rules_seg = run_fpgrowth_and_rules(
                    segment_baskets, 
                    min_support=MIN_SUPPORT_SEGMENT, 
                    name=f"Segment {int(c)}"
                )
                
                if not rules_seg.empty:
                    seg_path = os.path.join(OUTPUT_DIR, f"segment_{int(c)}_rules.csv")
                    rules_seg.to_csv(seg_path, index=False, encoding="utf-8-sig")
                    print(f"💾 Đã lưu Rules cho Segment {int(c)} tại: {seg_path}")
        else:
            print(f"⚠️ Không tìm thấy cột {TARGET_CLUSTER_COL} trong tệp phân cụm.")
    else:
        print(f"⚠️ Không tìm thấy tệp {CLUSTER_DATA_PATH}. Bỏ qua khai phá theo phân cụm.")
        
    print("\n🎉 Tiến trình Khai phá Luật kết hợp hoàn tất!")

if __name__ == "__main__":
    main()
