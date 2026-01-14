import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import json

# ---------------------------------------------------------------------
# IMPORT SUSPICIOUS BEHAVIOR FUNCTIONS
# ---------------------------------------------------------------------
from utils import *

# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------
logger = setup_logger("output/logs.log")


# ---------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Detect suspicious traders from temporal cycles")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to parquet dataset (same as cycle extraction)")
    parser.add_argument("--cycles-dir", type=str, default="cycles_data",
                        help="Directory containing cycle files from previous step")

    return parser.parse_args()


# ---------------------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    cycles_dir = Path(args.cycles_dir)
    output_dir = Path("output/results")
    automated_dir = Path("output/results/automated_trading")
    wash_trading_dir = Path("output/results/wash_trading")

    output_dir.mkdir(parents=True, exist_ok=True)
    automated_dir.mkdir(parents=True, exist_ok=True)
    wash_trading_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if not cycles_dir.exists():
        raise FileNotFoundError(cycles_dir)

    logger.info("Loading dataset")
    transactions = pd.read_parquet(dataset_path)

    logger.info("Loading Temporal Cycles")
    cycles = pd.read_parquet(cycles_dir)



    # -------------------------------------------------------------
    # Automated traders detection
    # -------------------------------------------------------------
    suspicious_automated_wallets = automated_trading_detection(cycles)
    suspicious_wallet_set = set(suspicious_automated_wallets.index) 
    suspicious_transactions = cycles[cycles["buyer"].isin(suspicious_wallet_set) ]
    df_suspicious_transactions = suspicious_transactions["sales"].explode().apply(pd.Series).reset_index(drop=True).drop_duplicates()
        
    suspicious_automated_wallets.to_csv(automated_dir / "automated_traders.csv")
    df_suspicious_transactions.to_csv(automated_dir / "automated_transactions.csv")

    logger.info(" Automated Traders Saved ")



    # -------------------------------------------------------------
    # Wash traders detection
    # -------------------------------------------------------------
    circular, stable_price, cross_nft = wash_trading_detection(cycles,max_workers=64)
    wash_trades = pd.concat([cross_nft, stable_price, circular], ignore_index=True, sort=False)
    wash_trades.drop(columns=['nft_id','price_stability','avg_jaccard_similarity','cross_nft_repetition','wallet_set'],inplace=True)
    suspicious_cycles = wash_trades.drop_duplicates(subset=['cycle_length','profit','buyer','collection_name','min_ts','max_ts'],inplace=False).copy()
    Wash_traders_same_collection = wallets_single_collection_activity(suspicious_cycles)
    suspicious_wash_traders = filter_wash_traders(Wash_traders_same_collection,transactions)
    wash_traders_wallets = set(suspicious_wash_traders['wallet'])
    
    wash_trading_tx = transactions[
        (transactions['seller_num'].isin(wash_traders_wallets)) |
        (transactions['buyer_num'].isin(wash_traders_wallets))].copy()
        
    suspicious_collections = flag_suspicious_collections(
        cycles,
        suspicious_cycles,
        transactions)

    suspicious_wash_traders.to_csv(wash_trading_dir / "wash_traders.csv")
    wash_trading_tx.to_csv(wash_trading_dir / "wash_trading_transactions.csv")
    with open(wash_trading_dir/"suspicious_collections.json", "w") as f:
        json.dump(list(suspicious_collections), f)

    logger.info(" Wash Traders Saved ")
    logger.info("Suspicious trading activity detection completed successfully")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
