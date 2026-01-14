from __future__ import annotations

import logging
from ast import literal_eval
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


"""

Suspicious automated trading activity detection.

This module detects potentially automated trading behavior using:
- Cycle repetition
- Timing consistency
- Rapid execution patterns

"""


# ------------------------------------------------------------------
# Default configuration (can be overridden via function parameters)
# ------------------------------------------------------------------

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_CYCLE_STD_SECONDS = 3600     # 1 hour
LOW_TIME_VARIATION_CV = 0.1    
RAPID_TX_SECONDS = 60




# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

def prepare_automated_data(
    df_cycles: pd.DataFrame,
    min_cycles: int = 2,
    date_format: str = DATE_FORMAT,
) -> pd.DataFrame:
    """
    Filter buyers with more than `min_cycles`, clean timestamps,
    and parse the sales column.
    """

    # Count cycles per buyer directly from df_cycles
    buyer_cycle_counts = (
        df_cycles
        .groupby("buyer")
        .size()
    )

    buyers = buyer_cycle_counts[
        buyer_cycle_counts > min_cycles
    ].index

    df = df_cycles.loc[df_cycles["buyer"].isin(buyers)].copy()

    # Convert timestamps
    df["min_ts"] = pd.to_datetime(df["min_ts"], format=date_format, errors="coerce")
    df["max_ts"] = pd.to_datetime(df["max_ts"], format=date_format, errors="coerce")

    # Parse sales column if needed
    if not df.empty and isinstance(df["sales"].iloc[0], str):
        tqdm.pandas(desc="Parsing sales")

        def _parse_sales(value):
            try:
                return literal_eval(value)
            except (ValueError, SyntaxError):
                return []

        df["sales"] = df["sales"].progress_apply(_parse_sales)

    return df



# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-based features for automation detection.
    """
    df = df.copy()
    tqdm.pandas(desc="Calculating time patterns")

    def calculate_time_differences(transactions: List[Dict]) -> List[float]:
        if transactions is None:
            return []
        try:
            if len(transactions) < 2:
                return []
            tx_sorted = sorted(transactions, key=lambda x: x["time"])
            times = [tx["time"] for tx in tx_sorted]
            return [(times[i + 1] - times[i]) / 1000 for i in range(len(times) - 1)]
        except Exception:
            return []

    df["time_differences"] = df["sales"].progress_apply(calculate_time_differences)
    df["cycle_duration"] = df["max_ts"] - df["min_ts"]

    df["time_diff_mean"] = df["time_differences"].apply(
        lambda x: np.mean(x) if x else 0
    )
    df["time_diff_std"] = df["time_differences"].apply(
        lambda x: np.std(x) if len(x) > 1 else 0
    )
    df["time_diff_cv"] = df.apply(
        lambda row: (
            row["time_diff_std"] / row["time_diff_mean"]
            if row["time_diff_mean"] > 0 else 0
        ),
        axis=1
    )

    return df


# ------------------------------------------------------------------
# Detection logic
# ------------------------------------------------------------------

def detect_suspicious_buyers(
    df: pd.DataFrame,
    max_cycle_std_seconds: int = MAX_CYCLE_STD_SECONDS,
    low_time_variation_cv: float = LOW_TIME_VARIATION_CV,
    rapid_tx_seconds: int = RAPID_TX_SECONDS,
) -> pd.DataFrame:
    """
    Aggregate per buyer and detect suspicious automation patterns.
    """

    def _buyer_patterns(group: pd.DataFrame) -> pd.Series:
        cycle_std_seconds = group["cycle_duration"].std().total_seconds()

        return pd.Series({
            "cycle_count": len(group),
            "consistent_duration": (
                cycle_std_seconds < max_cycle_std_seconds
            ),
            "low_time_variation": (
                group["time_diff_cv"] < low_time_variation_cv
            ).mean(),
            "rapid_transactions": (
                group["time_diff_mean"] <= rapid_tx_seconds
            ).mean(),
        })

    buyer_stats = df.groupby("buyer").apply(_buyer_patterns)

    suspicious = buyer_stats[
        (buyer_stats["consistent_duration"]) |
        (buyer_stats["low_time_variation"] > 0.9) |
        (buyer_stats["rapid_transactions"] > 0)
    ].sort_values("cycle_count", ascending=False)

    return suspicious


# ------------------------------------------------------------------
# Public pipeline
# ------------------------------------------------------------------

def automated_trading_detection(
    df_cycles: pd.DataFrame,
    min_cycles: int = 2,
    max_cycle_std_seconds: int = MAX_CYCLE_STD_SECONDS,
    low_time_variation_cv: float = LOW_TIME_VARIATION_CV,
    rapid_tx_seconds: int = RAPID_TX_SECONDS,
) -> pd.DataFrame:
    """
    End-to-end suspicious activity detection pipeline.
    """

    df_clean = prepare_automated_data(
        df_cycles,
        min_cycles=min_cycles,
    )

    df_features = add_time_features(df_clean)

    suspicious = detect_suspicious_buyers(
        df_features,
        max_cycle_std_seconds=max_cycle_std_seconds,
        low_time_variation_cv=low_time_variation_cv,
        rapid_tx_seconds=rapid_tx_seconds,
    )

    return suspicious


def wallets_transaction_features(table_buyer: pd.DataFrame) -> pd.DataFrame:
    """
    Build transaction count and NFT diversity features per wallet.
    """

    # ------------------------------
    # NFT diversity
    # ------------------------------
    buyer_nft = (
        table_buyer
        .groupby("buyer_num")["nft_num"]
        .nunique()
        .reset_index()
        .rename(columns={"buyer_num": "wallet", "nft_num": "unique_nfts_bought"})
    )

    seller_nft = (
        table_buyer
        .groupby("seller_num")["nft_num"]
        .nunique()
        .reset_index()
        .rename(columns={"seller_num": "wallet", "nft_num": "unique_nfts_sold"})
    )

    nft_features = buyer_nft.merge(
        seller_nft, on="wallet", how="left"
    ).fillna(0)

    # ------------------------------
    # Transaction counts (FIXED)
    # ------------------------------
    buyer_counts = (
        table_buyer
        .groupby("buyer_num")
        .size()
        .reset_index(name="num_bought")
        .rename(columns={"buyer_num": "wallet"})
    )

    seller_counts = (
        table_buyer
        .groupby("seller_num")
        .size()
        .reset_index(name="num_sold")
        .rename(columns={"seller_num": "wallet"})
    )

    tx_features = buyer_counts.merge(
        seller_counts, on="wallet", how="left"
    ).fillna(0)

    return tx_features.merge(nft_features, on="wallet", how="inner")


def wallets_temporal_features(table_buyer: pd.DataFrame) -> pd.DataFrame:
    """
    Build holding time and activity window features per wallet.
    """
    # Flatten transactions
    
    records = []
    for _, row in table_buyer.iterrows():
        records.append({
            "wallet": row["buyer_num"],
            "nft_id": row["nft_num"],
            "time": row["closing_date"],
            "role": "buyer",
        })
        records.append({
            "wallet": row["seller_num"],
            "nft_id": row["nft_num"],
            "time": row["closing_date"],
            "role": "seller",
        })

    df_tx = pd.DataFrame(records)
    df_tx["time"] = pd.to_datetime(df_tx["time"], errors="coerce")

    # Holding time (FIFO: first buy → first sell)


    
    hold_times = []
    for (wallet, nft_id), group in df_tx.groupby(["wallet", "nft_id"]):
        group = group.sort_values("time")
        buys = group[group["role"] == "buyer"]["time"].tolist()
        sells = group[group["role"] == "seller"]["time"].tolist()

        for b, s in zip(buys, sells):
            if s > b:
                hold_times.append({
                    "wallet": wallet,
                    "hold_time": s - b,
                })

    df_hold = pd.DataFrame(hold_times)
    if not df_hold.empty:
        df_hold = (
            df_hold
            .groupby("wallet")["hold_time"]
            .mean()
            .reset_index()
        )
    
    # Activity window
    df_long = pd.melt(
        table_buyer,
        id_vars=["closing_date"],
        value_vars=["buyer_num", "seller_num"],
        value_name="wallet",
    )

    df_activity = (
        df_long
        .groupby("wallet")["closing_date"]
        .agg(first_activity="min", last_activity="max")
        .reset_index()
    )

    df_activity["activity_period"] = (
        df_activity["last_activity"] - df_activity["first_activity"]
    )

    return df_activity.merge(df_hold, on="wallet", how="left")



def wallets_trading_details(
    suspicious_buyers: pd.DataFrame,
    transactions_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich suspicious wallets with transaction, NFT, holding-time,
    and activity-span features.
    """

    wallet_set = set(suspicious_buyers["buyer"])

    # Filter only relevant transactions
    table_buyer = transactions_table[
        transactions_table["buyer_num"].isin(wallet_set) |
        transactions_table["seller_num"].isin(wallet_set)
    ].copy()

    tx_features = wallets_transaction_features(table_buyer)
    time_features = wallets_temporal_features(table_buyer)
    enriched_features = tx_features.merge(
        time_features, on="wallet", how="left"
    )

    enriched = suspicious_buyers.merge(
        enriched_features,
        left_on="buyer",
        right_on="wallet",
        how="inner",
    )

    return enriched



import pandas as pd


def filter_automated_wallets(
    suspicious_wallets: pd.DataFrame,
    max_hold_time: pd.Timedelta = pd.Timedelta("1 day"),
    avg_daily_tx: float = 10.0,
) -> pd.DataFrame:
    """
    Filter suspicious wallets based on holding time, NFT repetition,
    and trading frequency.
    """

    df = suspicious_wallets.dropna().copy()

    # Ensure hold_time is timedelta
    df["hold_time"] = pd.to_timedelta(df["hold_time"])
    df["activity_period"] = pd.to_timedelta(df["activity_period"])

    # Average daily transactions
    df["average_daily_transaction"] = (
        df["num_bought"] + df["num_sold"]
    ) / (df["activity_period"].dt.days + 1)

    # Core behavioral filters
    df = df[df["hold_time"] < max_hold_time]
    df = df[df["unique_nfts_sold"] < df["num_sold"]]
    df = df[df["unique_nfts_bought"] < df["num_bought"]]

    # Frequency filters (two-stage, explicit)
    df = df[df["average_daily_transaction"] >= avg_daily_tx]

    return df
