import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from raphtory import Graph
from utils import temporal_cycles


# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------
def setup_logger():
    log_dir = Path("output/log")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"temporal_cycles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
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
    parser = argparse.ArgumentParser(description="Run temporal cycle Detection")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-duration", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-combo", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--num-processes", type=int, default=1,
                        help="Number of parallel processes (1 = sequential)")

    return parser.parse_args()


# ---------------------------------------------------------------------
# TEMPORAL GRAPH CONSTRUCTION
# ---------------------------------------------------------------------
def build_temporal_graph(df: pd.DataFrame) -> Graph:
    g = Graph()
    g.load_edges_from_pandas(
        df,
        src="seller_num",
        dst="buyer_num",
        time="closing_date",
        properties=["nft_num", "price_usd"],
        layer_col="chain",
    )
    return g


# ---------------------------------------------------------------------
# VALID GRAPH VIEW
# ---------------------------------------------------------------------
def valid_graph_view(rolling_g, max_duration: Optional[float] = None):
    if rolling_g is None or rolling_g.count_edges() == 0:
        logger.debug("Empty Temporal graph")
        return rolling_g

    valid_nodes = []

    for node in rolling_g.nodes:
        if node.in_degree() == 0 or node.out_degree() == 0:
            continue

        try:
            t_in = node.in_edges.earliest_time.min()
            t_out = node.out_edges.latest_time.max()
        except Exception:
            logger.debug(f"Invalid temporal data for node {node}")
            continue

        if t_out < t_in:
            continue

        if max_duration is not None and (t_out - t_in) > max_duration:
            continue

        valid_nodes.append(node.name)

    return rolling_g.subgraph(valid_nodes)


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
            rolling_g = valid_graph_view(rolling_g, args.max_duration)

            cycles = temporal_cycles(
                rolling_g,
                max_length=args.max_length,
                max_cycles=args.max_cycles,
                max_duration=args.max_duration,
                max_combo=args.max_combo,
            )

            if cycles:
                all_cycles.extend(list(cycles))

        if not all_cycles:
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

    output_dir = Path("cycles_data")
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
