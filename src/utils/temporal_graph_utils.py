from raphtory import Graph
from typing import Optional

import pandas as pd
import numpy as np


## plot motifs for the Gainers and loosers 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import distinctipy

from matplotlib.offsetbox import OffsetImage,AnnotationBbox

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
def valid_graph_view(rolling_g,logger,max_duration: Optional[float] = None):
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



def cdf(listlike, normalised=True):
    data = np.array(listlike)
    N = len(listlike)

    x = np.sort(data)
    if (normalised):
        y = np.arange(N)/float(N-1)
    else:
        y = np.arange(N)
    return x, y


def ccdf(listlike, normalised=True):
    x, y = cdf(listlike,normalised)
    if normalised:
        return x, 1.0-y
    else:
        return x, len(listlike)-y





# Mapping different motifs to their place in the heatmap.
mapper = {0:(5,5),1:(5,4),2:(4,5),3:(4,4),4:(4,3),5:(4,2),6:(5,3),7:(5,2),8:(0,0),9:(0,1),10:(1,0),11:(1,1),12:(2,1),13:(2,0),14:(3,1),15:(3,0),
          16:(0,5),17:(0,4),18:(1,5),19:(1,4),20:(2,3),21:(2,2),22:(3,3),23:(3,2),24:(5,0),25:(5,1),26:(4,0),27:(4,1),28:(4,1),29:(4,0),30:(5,1),
          31:(5,0),32:(0,2),33:(2,4),34:(1,2),35:(3,4),36:(0,3),37:(2,5),38:(1,3),39:(3,5)}

def to_3d_heatmap(motif_flat, data_type=int):
    motif_3d = np.zeros((6,6),dtype=data_type)
    for i in range(40):
        motif_3d[mapper[i]]=motif_flat[i]
    return motif_3d 