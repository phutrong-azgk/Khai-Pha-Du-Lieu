import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import hdbscan


# =========================================================
# AUTO BENCHMARK ENGINE
# =========================================================
def run_clustering_experiments(X_scaled, X_pca=None):

    results = []

    # =====================================================
    # 1. K-MEANS
    # =====================================================
    for k in range(2, 11):

        try:

            km = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = km.fit_predict(X_scaled)

            # =============================================
            # VALIDATE LABELS
            # =============================================
            unique_labels = np.unique(labels)

            if len(unique_labels) < 2:
                continue

            # =============================================
            # METRICS
            # =============================================
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)

            counts = np.bincount(labels)
            balance = np.min(counts) / np.max(counts)

            # =============================================
            # BONUS / PENALTY
            # =============================================
           # Ưu tiên mạnh số cụm lẻ
            odd_bonus = 0.12 if k % 2 == 1 else -0.05

            silhouette_bonus = (
                0.05 if sil > 0.45 else 0
            )

            imbalance_penalty = (
                -0.10 if balance < 0.15 else 0
            )

            cluster_penalty = -(k * 0.01)

            # =============================================
            # NORMALIZE CALINSKI
            # =============================================
            ch_norm = ch / 10000

            # =============================================
            # FINAL SCORE
            # =============================================
            final_score = (
                sil * 0.55
                - db * 0.15
                + ch_norm * 0.20
                + balance * 0.10
                + odd_bonus
                + silhouette_bonus
                + imbalance_penalty
                + cluster_penalty
            )

            # =============================================
            # ROUND FLOAT
            # =============================================
            sil = round(sil, 4)
            db = round(db, 4)
            ch = round(ch, 2)
            balance = round(balance, 4)
            final_score = round(final_score, 4)

            results.append({
                "Algorithm": "K-Means",
                "Params": f"k={k}",
                "K": k,
                "Silhouette": sil,
                "Davies": db,
                "Calinski": ch,
                "Balance": balance,
                "OddBonus": odd_bonus,
                "FinalScore": final_score
            })

        except Exception:
            continue

    # =====================================================
    # 2. GMM
    # =====================================================
    for c in range(2, 11):

        try:

            gmm = GaussianMixture(
                n_components=c,
                random_state=42
            )

            labels = gmm.fit_predict(X_scaled)

            unique_labels = np.unique(labels)

            if len(unique_labels) < 2:
                continue

            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)

            counts = np.bincount(labels)
            balance = np.min(counts) / np.max(counts)

            # Ưu tiên mạnh số cụm lẻ
            odd_bonus = 0.12 if c % 2 == 1 else -0.05

            silhouette_bonus = (
                0.05 if sil > 0.45 else 0
            )

            imbalance_penalty = (
                -0.10 if balance < 0.15 else 0
            )

            cluster_penalty = -(c * 0.01)

            ch_norm = ch / 10000

            final_score = (
                sil * 0.55
                - db * 0.15
                + ch_norm * 0.20
                + balance * 0.10
                + odd_bonus
                + silhouette_bonus
                + imbalance_penalty
                + cluster_penalty
            )

            sil = round(sil, 4)
            db = round(db, 4)
            ch = round(ch, 2)
            balance = round(balance, 4)
            final_score = round(final_score, 4)

            results.append({
                "Algorithm": "GMM",
                "Params": f"c={c}",
                "K": c,
                "Silhouette": sil,
                "Davies": db,
                "Calinski": ch,
                "Balance": balance,
                "OddBonus": odd_bonus,
                "FinalScore": final_score
            })

        except Exception:
            continue

    # =====================================================
    # 3. HDBSCAN
    # =====================================================
    try:

        hdb = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10
        )

        X_for_hdbscan = X_pca if X_pca is not None else X_scaled
        labels = hdb.fit_predict(X_for_hdbscan)

        valid_mask = labels != -1

        if valid_mask.sum() > 20:

            clean_labels = labels[valid_mask]
            clean_X = X_for_hdbscan[valid_mask]

            unique_labels = np.unique(clean_labels)

            if len(unique_labels) > 1:

                sil = silhouette_score(
                    clean_X,
                    clean_labels
                )

                db = davies_bouldin_score(
                    clean_X,
                    clean_labels
                )

                ch = calinski_harabasz_score(
                    clean_X,
                    clean_labels
                )

                counts = np.bincount(clean_labels)

                balance = (
                    np.min(counts) / np.max(counts)
                )

                k_hdb = len(unique_labels)

                # Ưu tiên mạnh số cụm lẻ
                odd_bonus = (
                    0.12 if k_hdb % 2 == 1 else -0.05
                )

                silhouette_bonus = (
                    0.05 if sil > 0.45 else 0
                )

                imbalance_penalty = (
                    -0.10 if balance < 0.15 else 0
                )

                cluster_penalty = (
                    -(k_hdb * 0.01)
                )

                ch_norm = ch / 10000

                final_score = (
                    sil * 0.55
                    - db * 0.15
                    + ch_norm * 0.20
                    + balance * 0.10
                    + odd_bonus
                    + silhouette_bonus
                    + imbalance_penalty
                    + cluster_penalty
                )

                sil = round(sil, 4)
                db = round(db, 4)
                ch = round(ch, 2)
                balance = round(balance, 4)
                final_score = round(final_score, 4)

                results.append({
                    "Algorithm": "HDBSCAN",
                    "Params": "dynamic",
                    "K": k_hdb,
                    "Silhouette": sil,
                    "Davies": db,
                    "Calinski": ch,
                    "Balance": balance,
                    "OddBonus": odd_bonus,
                    "FinalScore": final_score
                })

    except Exception:
        pass

    # =====================================================
    # CREATE DATAFRAME
    # =====================================================
    df = pd.DataFrame(results)

    if df.empty:
        return df

    # =====================================================
    # SORT BY FINAL SCORE
    # =====================================================
    df = df.sort_values(
        by="FinalScore",
        ascending=False
    ).reset_index(drop=True)

    return df


# =========================================================
# SMART RECOMMENDATION ENGINE
# =====================================================
def get_recommendation(results_df):

    if results_df.empty:
        return "Không đủ dữ liệu để đề xuất."

    best = results_df.iloc[0]

    return (
        f"Đề xuất tối ưu: "
        f"{best['Algorithm']} ({best['Params']}) "
        f"| FinalScore={best['FinalScore']:.3f} "
        f"| Silhouette={best['Silhouette']:.3f} "
        f"| Davies={best['Davies']:.3f} "
        f"| Balance={best['Balance']:.2f}"
    )