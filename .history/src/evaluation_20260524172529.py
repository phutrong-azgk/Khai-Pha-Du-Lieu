import os
import logging
import numpy as np
import pandas as pd
import joblib

from scipy.stats import entropy, rankdata

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# ==========================================================
# LOGGING CONFIG
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================================
# FEATURE CONFIG
# ==========================================================
# MUST MATCH customer_features.csv
FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Interpurchase_Time",
    "Brand_Diversity",
    "Category_Diversity",
    "Average_Basket_Value"
]

# ==========================================================
# PIPELINE CONFIG
# ==========================================================
PIPELINE_CONFIG = {

    "evaluation": {

        "sample_size": 5000,

        "random_state": 42,

        # Maximum clusters allowed
        "max_business_k": 10,

        "weights": {

            "silhouette": 0.30,

            "db_inverse": 0.20,

            "ch": 0.15,

            "balance": 0.20,

            "stability": 0.15
        },

        "penalties": {

            # Penalize K=2 lazy segmentation
            "k_lazy": 2,

            "k_lazy_value": 0.10,

            # Penalize low entropy distributions
            "low_entropy_threshold": 0.40,

            "low_entropy_value": 0.20,

            # Penalize excessive noise
            "high_noise_threshold": 0.35,

            "high_noise_value": 0.25
        }
    },

    "persona": {

        "quantiles": {

            "high": 0.75,

            "low": 0.25
        }
    }
}

# ==========================================================
# PERCENTILE NORMALIZATION
# ==========================================================
def percentile_rank(series: pd.Series):

    n = len(series)

    if n <= 1:
        return np.ones(n)

    ranks = rankdata(series, method="average")

    return (ranks - 1) / (n - 1)

# ==========================================================
# METRIC ENGINE
# ==========================================================
def calc_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    config: dict
):

    labels = np.asarray(labels)

    # Remove HDBSCAN noise
    mask = labels != -1

    X_eval = X_scaled[mask]

    y_eval = labels[mask]

    n_clusters = len(set(y_eval))

    if n_clusters < 2:
        return None

    # ======================================================
    # CLUSTER DISTRIBUTION
    # ======================================================
    _, counts = np.unique(y_eval, return_counts=True)

    cluster_pcts = counts / len(y_eval)

    balance_score = float(
        counts.min() / counts.max()
    )

    cluster_entropy = (
        entropy(counts) / np.log(n_clusters)
        if n_clusters > 1 else 0.0
    )

    # ======================================================
    # SAFE SUBSAMPLING
    # ======================================================
    sample_size = config["sample_size"]

    if len(X_eval) > sample_size:

        np.random.seed(config["random_state"])

        idx = np.random.choice(
            len(X_eval),
            sample_size,
            replace=False
        )

        X_sub = X_eval[idx]

        y_sub = y_eval[idx]

        # Safety fallback
        if len(set(y_sub)) < 2:

            X_sub = X_eval

            y_sub = y_eval

    else:

        X_sub = X_eval

        y_sub = y_eval

    # ======================================================
    # METRICS
    # ======================================================
    sil = silhouette_score(X_sub, y_sub)

    db = davies_bouldin_score(X_sub, y_sub)

    ch = calinski_harabasz_score(X_sub, y_sub)

    return {

        "N_Clusters": n_clusters,

        "Silhouette": float(sil),

        "Davies_Bouldin": float(db),

        "Calinski_Harabasz": float(ch),

        "Balance_Score": float(balance_score),

        "Cluster_Entropy": float(cluster_entropy),

        "Min_Cluster_Pct": float(cluster_pcts.min()),

        "Max_Cluster_Pct": float(cluster_pcts.max())
    }

# ==========================================================
# MODEL EVALUATION
# ==========================================================
def evaluate_all(
    X_scaled: np.ndarray,
    clustered: pd.DataFrame,
    config: dict,
    output_path: str = "output/evaluation_results.csv"
):

    rows = []

    eval_cfg = config["evaluation"]

    # ======================================================
    # KMEANS + GMM
    # ======================================================
    for prefix, algo_name in [

        ("k", "K-Means"),

        ("g", "GMM")

    ]:

        for k in range(2, 10):

            col = f"{prefix}{k}"

            if col not in clustered.columns:
                continue

            metrics = calc_metrics(
                X_scaled,
                clustered[col].values,
                eval_cfg
            )

            if metrics is None:
                continue

            rows.append({

                "Algorithm": algo_name,

                "Config_K": k,

                "Noise_Ratio": 0.0,

                "HDBSCAN_Stability": 1.0,

                **metrics
            })

    # ======================================================
    # HDBSCAN
    # ======================================================
    hdb_col = next(

        (
            c for c in ["hdb", "cluster_hdbscan"]
            if c in clustered.columns
        ),

        None
    )

    if hdb_col is not None and "hdb_persistence" in clustered.columns:

        labels = clustered[hdb_col].values

        noise_ratio = np.mean(labels == -1)

        valid_hdb = clustered[
            clustered[hdb_col] != -1
        ]["hdb_persistence"]

        median_persistence = (
            valid_hdb.median()
            if not valid_hdb.empty
            else 0.0
        )

        metrics = calc_metrics(
            X_scaled,
            labels,
            eval_cfg
        )

        if metrics is not None:

            rows.append({

                "Algorithm": "HDBSCAN",

                "Config_K": np.nan,

                "Noise_Ratio": round(
                    float(noise_ratio), 4
                ),

                "HDBSCAN_Stability": float(
                    median_persistence
                ),

                **metrics
            })

    # ======================================================
    # RESULTS TABLE
    # ======================================================
    results = pd.DataFrame(rows)

    if results.empty:
        raise ValueError(
            "No clustering results found."
        )

    # ======================================================
    # DB INVERSE
    # ======================================================
    results["DB_Inverse"] = (
        1 / (1 + results["Davies_Bouldin"])
    )

    # ======================================================
    # PERCENTILE NORMALIZATION
    # ======================================================
    metrics_to_scale = [

        "Silhouette",

        "DB_Inverse",

        "Calinski_Harabasz",

        "Balance_Score",

        "HDBSCAN_Stability"
    ]

    for col in metrics_to_scale:

        results[f"norm_{col}"] = percentile_rank(
            results[col]
        )

    # ======================================================
    # FINAL SCORE
    # ======================================================
    def compute_final_score(row):

        w = eval_cfg["weights"]

        p = eval_cfg["penalties"]

        base_score = (

            row["norm_Silhouette"]
            * w["silhouette"]

            +

            row["norm_DB_Inverse"]
            * w["db_inverse"]

            +

            row["norm_Calinski_Harabasz"]
            * w["ch"]

            +

            row["norm_Balance_Score"]
            * w["balance"]

            +

            row["norm_HDBSCAN_Stability"]
            * w["stability"]
        )

        penalty = 0.0

        k = row["N_Clusters"]

        # ==================================================
        # MAX BUSINESS K
        # ==================================================
        if k > eval_cfg["max_business_k"]:

            penalty += 0.20

        # ==================================================
        # K=2 PENALTY
        # ==================================================
        if k == p["k_lazy"]:

            penalty += p["k_lazy_value"]

        # ==================================================
        # LOW ENTROPY PENALTY
        # ==================================================
        if row["Cluster_Entropy"] < p["low_entropy_threshold"]:

            penalty += p["low_entropy_value"]

        # ==================================================
        # HIGH NOISE PENALTY
        # ==================================================
        if row["Noise_Ratio"] > p["high_noise_threshold"]:

            penalty += p["high_noise_value"]

        return round(

            max(0.0, float(base_score - penalty)),

            4
        )

    results["Final_Score"] = results.apply(
        compute_final_score,
        axis=1
    )

    # ======================================================
    # BEST MODEL
    # ======================================================
    best_idx = results["Final_Score"].idxmax()

    results["Best"] = False

    results.loc[best_idx, "Best"] = True

    best_row = results.loc[best_idx]

    logging.info(

        f"Selected Model: "

        f"{best_row['Algorithm']} "

        f"(K={int(best_row['N_Clusters'])}) "

        f"| Score={best_row['Final_Score']:.4f}"
    )

    # ======================================================
    # SAVE
    # ======================================================
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    results.to_csv(output_path, index=False)

    return (

        results,

        best_row["Algorithm"],

        best_row["Config_K"]
    )

# ==========================================================
# PERSONA ENGINE
# ==========================================================
def build_personas_matrix(
    features: pd.DataFrame,
    clustered: pd.DataFrame,
    best_algo: str,
    best_k,
    config: dict,
    output_path: str = "output/customer_personas.csv"
):

    # ======================================================
    # LABEL COLUMN
    # ======================================================
    if best_algo == "K-Means":

        label_col = f"k{int(best_k)}"

    elif best_algo == "GMM":

        label_col = f"g{int(best_k)}"

    else:

        label_col = (
            "hdb"
            if "hdb" in clustered.columns
            else "cluster_hdbscan"
        )

    # ======================================================
    # MERGE
    # ======================================================
    merge_cols = ["user_id", label_col]

    if "outlier_scores" in clustered.columns:
        merge_cols.append("outlier_scores")

    merged = features.merge(
        clustered[merge_cols],
        on="user_id"
    )

    merged.rename(
        columns={label_col: "cluster"},
        inplace=True
    )

    # ======================================================
    # GLOBAL REFERENCES
    # ======================================================
    feature_cols = FEATURE_COLS.copy()

    global_median = features[
        feature_cols
    ].median()

    global_q75 = features[
        feature_cols
    ].quantile(
        config["persona"]["quantiles"]["high"]
    )

    global_q25 = features[
        feature_cols
    ].quantile(
        config["persona"]["quantiles"]["low"]
    )

    # ======================================================
    # ROBUST AGGREGATION
    # ======================================================
    stats = merged.groupby(
        "cluster"
    )[feature_cols].median().reset_index()

    # ======================================================
    # OUTLIER SCORE
    # ======================================================
    if "outlier_scores" in merged.columns:

        outlier_stats = merged.groupby(
            "cluster"
        )["outlier_scores"].median().reset_index()

        stats = stats.merge(
            outlier_stats,
            on="cluster"
        )

    else:

        stats["outlier_scores"] = 0.0

    # ======================================================
    # CLUSTER SIZE
    # ======================================================
    cluster_counts = merged.groupby(
        "cluster"
    ).size().reset_index(name="count")

    stats = stats.merge(
        cluster_counts,
        on="cluster"
    )

    stats["pct"] = (
        stats["count"]
        / stats["count"].sum()
        * 100
    ).round(1)

    # ======================================================
    # PERSONA TAGS
    # ======================================================
    def generate_tags(row):

        cluster_id = row["cluster"]

        # ==================================================
        # OUTLIERS
        # ==================================================
        if cluster_id == -1:

            if row["outlier_scores"] > 0.75:
                return ["Hard Anomaly"]

            return ["Soft Outlier"]

        tags = []

        rec = row["Recency"]

        freq = row["Frequency"]

        mon = row["Monetary"]

        div = row["Category_Diversity"]

        inter = row["Interpurchase_Time"]

        basket = row["Average_Basket_Value"]

        # ==================================================
        # VALUE TIERS
        # ==================================================
        if (

            mon >= global_q75["Monetary"]

            and

            freq >= global_q75["Frequency"]
        ):

            tags.append("Elite VIP")

        elif (

            mon >= global_median["Monetary"]

            and

            freq >= global_median["Frequency"]
        ):

            tags.append("Core Regular")

        elif mon <= global_q25["Monetary"]:

            tags.append("Low Value")

        # ==================================================
        # ENGAGEMENT
        # ==================================================
        if rec <= global_q25["Recency"]:

            tags.append("High Active")

        elif rec >= global_q75["Recency"]:

            tags.append("Risk of Churn")

        # ==================================================
        # BUYING BEHAVIOR
        # ==================================================
        if basket >= global_q75["Average_Basket_Value"]:

            tags.append("Big Basket Buyer")

        if inter <= global_q25["Interpurchase_Time"]:

            tags.append("Frequent Shopper")

        if div >= global_q75["Category_Diversity"]:

            tags.append("Category Explorer")

        # ==================================================
        # FALLBACK
        # ==================================================
        if not tags:

            tags.append("General Audience")

        return tags

    stats["Persona_Tags"] = stats.apply(
        generate_tags,
        axis=1
    )

    stats["Persona_Display"] = stats[
        "Persona_Tags"
    ].apply(
        lambda x: " | ".join(x)
    )

    # ======================================================
    # SAVE
    # ======================================================
    ordered_cols = [

        "cluster",

        "Persona_Display",

        "count",

        "pct",

        "outlier_scores"

    ] + feature_cols

    stats = stats[ordered_cols]

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    stats.to_csv(output_path, index=False)

    logging.info(
        f"Persona matrix saved: {output_path}"
    )

    return stats

# ==========================================================
# MAIN PIPELINE
# ==========================================================
if __name__ == "__main__":

    logging.info(
        "Starting clustering evaluation pipeline..."
    )

    try:

        # ==================================================
        # LOAD FILES
        # ==================================================
        features = pd.read_csv(
            "output/customer_features.csv"
        )

        clustered = pd.read_csv(
            "output/clustered_customers.csv"
        )

        scaler = joblib.load(
            "models/scaler.pkl"
        )

        # ==================================================
        # FEATURE VALIDATION
        # ==================================================
        if hasattr(scaler, "feature_names_in_"):

            expected_features = list(
                scaler.feature_names_in_
            )

            missing_feats = [

                f for f in expected_features
                if f not in features.columns
            ]

            if missing_feats:

                raise ValueError(

                    f"Missing required features: "

                    f"{missing_feats}"
                )

            X = features[
                expected_features
            ].values

        else:

            logging.warning(
                "Scaler metadata missing. "
                "Using FEATURE_COLS fallback."
            )

            missing_feats = [

                f for f in FEATURE_COLS
                if f not in features.columns
            ]

            if missing_feats:

                raise ValueError(

                    f"Missing required features: "

                    f"{missing_feats}"
                )

            X = features[
                FEATURE_COLS
            ].values

        # ==================================================
        # SCALE
        # ==================================================
        X_scaled = scaler.transform(X)

        # ==================================================
        # EVALUATE MODELS
        # ==================================================
        results, best_algo, best_k = evaluate_all(

            X_scaled=X_scaled,

            clustered=clustered,

            config=PIPELINE_CONFIG
        )

        # ==================================================
        # BUILD PERSONAS
        # ==================================================
        personas_matrix = build_personas_matrix(

            features=features,

            clustered=clustered,

            best_algo=best_algo,

            best_k=best_k,

            config=PIPELINE_CONFIG
        )

        logging.info(
            "Pipeline executed successfully."
        )

    except Exception as e:

        logging.exception(
            f"Pipeline failed: {e}"
        )