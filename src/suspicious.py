import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------
# IMPORT SUSPICIOUS BEHAVIOR FUNCTIONS
# ---------------------------------------------------------------------
# Adjust the import path if needed
from utils import automated_traders, wash_traders


# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------
def setup_logger():
    log_dir = Path("output/log")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"suspicious_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )

    return logging.getLogger(__name__)


logger = setup_logger()


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

    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if not cycles_dir.exists():
        raise FileNotFoundError(cycles_dir)

    logger.info("Loading dataset")
    df = pd.read_parquet(dataset_path)

    if "collection_name" not in df.columns:
        raise ValueError("Dataset must contain a 'collection_name' column")

    cycle_files = list(cycles_dir.glob("*_cycles.pkl"))
    logger.info(f"Found {len(cycle_files)} cycle files")

    for cycle_file in tqdm(cycle_files, desc="Processing collections"):
        collection = cycle_file.stem.replace("_cycles", "")
        logger.info(f"Processing collection: {collection}")

        collection_df = df[df["collection_name"] == collection]
        cycles = pd.read_pickle(cycle_file)

        if not cycles:
            logger.info(f"No cycles for collection {collection}")
            continue

        # -------------------------------------------------------------
        # Automated traders detection
        # -------------------------------------------------------------
        automated_result = automated_traders(
            collection_df=collection_df,
            cycles=cycles,
        )

        automated_out = output_dir / f"{collection}_automated_traders.pkl"
        pd.to_pickle(automated_result, automated_out)

        logger.info(f"Saved automated traders for {collection}")

        # -------------------------------------------------------------
        # Wash traders detection
        # -------------------------------------------------------------
        wash_result = wash_traders(
            collection_df=collection_df,
            cycles=cycles,
        )

        wash_out = output_dir / f"{collection}_wash_traders.pkl"
        pd.to_pickle(wash_result, wash_out)

        logger.info(f"Saved wash traders for {collection}")

    logger.info("Suspicious trader detection completed successfully")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
