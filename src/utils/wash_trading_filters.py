import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

from typing import Set


# -------------------------------------------------------
# Price stability
# -------------------------------------------------------
def compute_price_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'price_stability' = std(prices) / mean(prices)
    """
    def _stability(sales):
        prices = [tx["price_usd"] for tx in sales]
        if len(prices) < 2:
            return 0.0

        mean_price = np.mean(prices)
        if mean_price == 0:
            return 0.0

        return np.std(prices) / mean_price

    df = df.copy()
    df["price_stability"] = df["sales"].apply(_stability)
    return df


# -------------------------------------------------------
# Wallet set extraction
# -------------------------------------------------------
def extract_wallet_sets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'wallet_set' column per cycle.
    """
    df = df.copy()
    df["wallet_set"] = df["sales"].apply(
        lambda sales: set(
            [tx["buyer"] for tx in sales] +
            [tx["seller"] for tx in sales]
        )
    )
    return df



def process_collection(collection_name, group_df, threshold=0.5):
    """
    Compute repetition metrics for a single collection using *exact* Jaccard similarity.
    Returns: list of (index, repetition_count, avg_similarity)
    """
    group_df = group_df.copy()
    wallet_sets = group_df["wallet_set"].to_dict()

    results = []
    for i, wallet_set_i in wallet_sets.items():
        count_repeats = 0
        sims = []

        for j, wallet_set_j in wallet_sets.items():
            if i == j:
                continue

            # Compute exact Jaccard similarity
            intersection = len(wallet_set_i & wallet_set_j)
            union = len(wallet_set_i | wallet_set_j)
            if union == 0:
                continue
            jaccard_similarity = intersection / union

            if jaccard_similarity >= threshold:
                count_repeats += 1
                sims.append(jaccard_similarity)

        avg_sim = sum(sims) / len(sims) if sims else 0.0
        results.append((i, count_repeats, avg_sim))

    return results


def compute_cross_nft_repetition_parallel(df, threshold=0.5, max_workers=None):
    """
    Fully parallel, exact Jaccard similarity computation.
    One process per collection.
    """
    # Extract collection names once
    df = df[(df['cycle_length']>2)].copy()
    df["collection_name"] = df["sales"].apply(lambda s: s[0]["collection_name"])
    groups = list(df.groupby("collection_name"))

    repetition_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_collection, name, group, threshold): name
            for name, group in groups
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing collections (parallel, exact Jaccard)"):
            repetition_data.extend(future.result())

    # Merge results back into DataFrame
    rep_map_count = {idx: count for idx, count, _ in repetition_data}
    rep_map_avg = {idx: avg for idx, _, avg in repetition_data}

    df["cross_nft_repetition"] = df.index.map(rep_map_count.get)
    df["avg_jaccard_similarity"] = df.index.map(rep_map_avg.get)

    return df


# -------------------------------------------------------
# Flagging logic (MAIN ENTRY POINT)
# -------------------------------------------------------
def wash_trading_detection(
    df_cycles: pd.DataFrame,
    price_stability_threshold: float = 0.05,
    jaccard_threshold: float = 0.666,
    max_workers: int | None = None,
):
    """
    Detect wash trading anomalies and return them separately.

    Returns
    -------
    circular_trades : pd.DataFrame
    stable_price_cycles : pd.DataFrame
    cross_nft_repetition_cycles : pd.DataFrame
    """

    df = df_cycles.copy()

    # -------------------------------
    # 1. Circular trading
    # -------------------------------
    df["flag_circular_trade"] = df["sales"].apply(
        lambda txs: len({tx["nft_id"] for tx in txs}) == 1
    )

    circular_trades = df[df["flag_circular_trade"]].copy()
    circular_trades["nft_id"] = circular_trades["sales"].apply(
        lambda txs: txs[0]["nft_id"] if len(txs) > 0 else None)
    circular_trades.drop_duplicates(subset=['cycle_length','profit','buyer','collection_name','min_ts','nft_id'],inplace=True)
   
    # -------------------------------
    # 2. Stable price cycles
    # -------------------------------
    if "price_stability" not in df.columns:
        df = compute_price_stability(df)

    df["flag_stable_price"] = (
        (df["price_stability"] > 0) &
        (df["price_stability"] < price_stability_threshold)
    )

    stable_price_cycles = df[df["flag_stable_price"]].copy()
    stable_price_cycles = stable_price_cycles[stable_price_cycles['cycle_length']>2]

    # -------------------------------
    # 3. Cross-NFT wallet repetition
    # -------------------------------
    if "wallet_set" not in df.columns:
        df = extract_wallet_sets(df)

    if "cross_nft_repetition" not in df.columns:
        df = compute_cross_nft_repetition_parallel(
            df,
            threshold=jaccard_threshold,
            max_workers=max_workers,
        )

    df["flag_cross_nft_repetition"] = df["cross_nft_repetition"] > 0

    cross_nft_repetition_cycles = df[
        df["flag_cross_nft_repetition"]
    ].copy()
    cross_nft_repetition_cycles = cross_nft_repetition_cycles[cross_nft_repetition_cycles['cross_nft_repetition']>2]

    return (
        circular_trades,
        stable_price_cycles,
        cross_nft_repetition_cycles,
    )




def wallets_single_collection_activity(
    wash_trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify wallets that appear in only one collection and count
    their occurrences per collection.

    Expected columns:
    - wallets (list-like)
    - collection_name
    """

    # Explode wallets so each wallet is on its own row
    wash_trades_ = wash_trades.drop_duplicates(subset=['cycle_length','profit','buyer','collection_name','min_ts']).copy()
    df_exploded = wash_trades_.explode("wallets")

    # Count how many collections each wallet appears in
    wallet_collection_counts = (
        df_exploded.groupby("wallets")["collection_name"]
        .nunique()
    )

    # Wallets appearing in exactly one collection
    single_collection_wallets = wallet_collection_counts[
        wallet_collection_counts == 1
    ].index

    # Filter to keep only those wallets
    filtered = df_exploded[df_exploded["wallets"].isin(single_collection_wallets)]

    # Count appearances per wallet per collection
    result = (
        filtered.groupby(["collection_name", "wallets"])
        .size()
        .reset_index(name="count")
    )

    return result[result['count']>2]




def filter_wash_traders(
    wash_traders_same_collection: pd.DataFrame,
    transactions: pd.DataFrame,
    min_in_group_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Identify wallets engaged in wash trading within the same collection.

    Returns wallet-level statistics with filtering applied.
    """

    # Wallets involved in suspicious cycles
    cycle_wallets = set(wash_traders_same_collection["wallets"])

    # Relevant transactions
    df_tx = transactions[
        (transactions["seller_num"].isin(cycle_wallets)) |
        (transactions["buyer_num"].isin(cycle_wallets))
    ].copy()

    # -----------------------------
    # Total transactions per wallet
    # -----------------------------
    total_counts = pd.concat(
        [df_tx["seller_num"], df_tx["buyer_num"]]
    ).value_counts().rename("total_transactions")

    # -----------------------------
    # In-group transactions
    # -----------------------------
    in_group_tx = df_tx[
        (df_tx["seller_num"].isin(cycle_wallets)) &
        (df_tx["buyer_num"].isin(cycle_wallets))
    ]

    in_group_counts = pd.concat(
        [in_group_tx["seller_num"], in_group_tx["buyer_num"]]
    ).value_counts().rename("in_group_transactions")

    # -----------------------------
    # Unique counterparties
    # -----------------------------
    pairs = pd.concat([
        df_tx[["seller_num", "buyer_num"]],
        df_tx[["buyer_num", "seller_num"]]
            .rename(columns={"buyer_num": "seller_num", "seller_num": "buyer_num"})
    ])

    unique_neighbors = (
        pairs.groupby("seller_num")["buyer_num"]
        .nunique()
        .rename("unique_counterparties")
    )

    # -----------------------------
    # Build wallet stats table
    # -----------------------------
    wallet_stats = pd.concat(
        [total_counts, in_group_counts, unique_neighbors],
        axis=1
    ).fillna(0)

    wallet_stats["in_group_ratio"] = (
        wallet_stats["in_group_transactions"] / wallet_stats["total_transactions"]
    )

    wallet_stats["wallet"] = wallet_stats.index

    # -----------------------------
    # Apply filters (your logic)
    # -----------------------------
    flagged_wash_traders = wallet_stats[
        (wallet_stats["wallet"].isin(cycle_wallets)) &
        (wallet_stats["in_group_ratio"] > min_in_group_ratio) &
        (wallet_stats["total_transactions"] > wallet_stats["unique_counterparties"])
    ].sort_values("in_group_ratio", ascending=False)

    return flagged_wash_traders.reset_index(drop=True)



def flag_suspicious_collections(
    df_cycles: pd.DataFrame,
    suspicious_cycles: pd.DataFrame,
    transactions: pd.DataFrame,
    min_wallet_participation: int = 2,
    min_total_price_usd: float = 1_000_000,
    min_row_count: int = 100,
    min_cycle_count: int = 2,
    min_unique_nfts: int = 100,
) -> Set[str]:
    """
    Identify suspicious NFT collections based on circular trading wallets.

    Returns
    -------
    Set[str]
        Set of flagged suspicious collection names
    """

    # -----------------------------
    # 1. Count cycles per collection
    # -----------------------------
    df_cycle_count = (
        df_cycles["collection_name"]
        .value_counts()
        .reset_index()
    )
    df_cycle_count.columns = ["collection_name", "cycle_count"]

    # -----------------------------
    # 2. Wallet participation in cycles
    # -----------------------------
    df_exploded = suspicious_cycles.explode("wallets")

    wallet_counts = (
        df_exploded["wallets"]
        .value_counts()
        .reset_index()
    )
    wallet_counts.columns = ["wallet", "participation_count"]
    wallet_counts["participation_count"] = wallet_counts["participation_count"].astype(int)

    wallet_counts = wallet_counts[
        wallet_counts["participation_count"] > min_wallet_participation
    ]

    cycle_wallets = set(wallet_counts["wallet"])

    if not cycle_wallets:
        return set()

    # -----------------------------
    # 3. Transactions involving cycle wallets
    # -----------------------------
    df_cycle_tx = transactions.loc[
        (transactions["seller_num"].isin(cycle_wallets)) |
        (transactions["buyer_num"].isin(cycle_wallets))
    ].drop_duplicates()

    # -----------------------------
    # 4. Collection statistics
    # -----------------------------
    df_stats = (
        df_cycle_tx
        .groupby("collection_name")
        .agg(
            row_count=("collection_name", "size"),
            total_price_usd=("price_usd", "sum"),
        )
        .reset_index()
    )

    df_unique = (
        df_cycle_tx
        .groupby("collection_name")["nft_num"]
        .nunique()
        .reset_index(name="unique_nfts")
    )

    # -----------------------------
    # 5. Merge all metrics
    # -----------------------------
    df_final = (
        df_stats
        .merge(df_cycle_count, on="collection_name", how="left")
        .merge(df_unique, on="collection_name", how="left")
        .fillna(0)
    )

    # -----------------------------
    # 6. Apply flagging rules
    # -----------------------------
    flagged_collections = df_final.loc[
        (df_final["total_price_usd"] > min_total_price_usd) &
        (df_final["row_count"] > min_row_count) &
        (df_final["cycle_count"] > min_cycle_count) &
        (df_final["unique_nfts"] > min_unique_nfts),
        "collection_name"
    ]

    return set(flagged_collections)
