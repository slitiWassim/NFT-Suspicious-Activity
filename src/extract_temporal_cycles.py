import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


from utils import temporal_cycles , build_temporal_graph , valid_graph_view , setup_logger


# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------
logger = setup_logger("output/logs.log")


# ---------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run temporal cycle Detection")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-duration", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-combo", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--window", type=str, required=True)
    parser.add_argument("--step", type=str, required=True)
    parser.add_argument("--num-processes", type=int, default=1,
                        help="Number of parallel processes (1 = sequential)")

    return parser.parse_args()



def parse_duration(duration: str) -> int:
    value, unit = duration.split()

    value = int(value)
    unit = unit.lower()

    if unit in ["hour", "hours"]:
        return int(36e5 * value)
    elif unit in ["day", "days"]:
        return int(36e5 * 24 * value)
    elif unit in ["minute", "minutes"]:
        return int(36e5 / 60 * value)
    else:
        raise ValueError(f"Unsupported duration unit: {unit}")


# ---------------------------------------------------------------------
# PER-COLLECTION PROCESSING (MULTIPROCESS SAFE)
# ---------------------------------------------------------------------
def process_collection(
    collection: str,
    collection_df: pd.DataFrame,
    args,
    output_dir: Path,
):
    try:
        g = build_temporal_graph(collection_df)
        all_cycles = []
        for rolling_g in g.rolling(window=args.window, step=args.step):
            rolling_g = valid_graph_view(rolling_g,logger,parse_duration(args.max_duration))

            cycles = temporal_cycles(
                rolling_g,
                max_length=args.max_length,
                max_cycles=args.max_cycles,
                max_duration=parse_duration(args.max_duration),
                max_combo=args.max_combo,
            )

            
            all_cycles += list(cycles)

        if len(all_cycles)==0 :
            logger.info(f"No cycles found for collection {collection}")
            return

        # Deduplicate
        all_cycles = list(set((tuple(w), tuple(t)) for w, t in all_cycles))

        output_file = output_dir / f"{collection}_cycles.pkl"
        pd.to_pickle(all_cycles, output_file)

        logger.info(f"Saved {len(all_cycles)} cycles for {collection}")

    except Exception as e:
        logger.exception(f"Failed processing collection {collection}: {e}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    output_dir = Path("data/cycles_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    if "collection_name" not in df.columns:
        raise ValueError("Missing 'collection_name' column")

    collections = df["collection_name"].unique()
    logger.info(f"Found {len(collections)} collections")

    # Sequential
    if args.num_processes == 1:
        for collection in tqdm(collections, desc="Processing collections"):
            collection_df = df[df["collection_name"] == collection]
            process_collection(collection, collection_df, args, output_dir)

    # Parallel
    else:
        logger.info(f"Running with {args.num_processes} processes")
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            futures = [
                executor.submit(
                    process_collection,
                    collection,
                    df[df["collection_name"] == collection],
                    args,
                    output_dir,
                )
                for collection in collections
            ]

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing collections"):
                pass

    logger.info("All collections processed successfully")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
