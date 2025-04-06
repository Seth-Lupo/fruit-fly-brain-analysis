#df schema 

#           pre_pt_root_id     post_pt_root_id neuropil  syn_count nt_type
# 0         720575940613354467  720575940616690107     ME_L          1     ACH
# 1         720575940625363947  720575940623224444     ME_L         13    GABA
# 2         720575940630432382  720575940618518557     ME_L         71     ACH
# 3         720575940627314521  720575940626337738     ME_L         10     NaN
# ...

import numpy as np
import pandas as pd
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG, WARNING, etc.
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logging.info("Loading data...")
path = "./connections_princeton_no_threshold.csv"
df = pd.read_csv(path)
logging.info("Finished loading data.")


# Build adjacency list from df
logging.info("Creating adjacency list...")
adj = df.groupby("pre_pt_root_id")["post_pt_root_id"].apply(list).to_dict()
logging.info("Finished creating adjacency list...")

# BFS to collect a local subgraph of row indices
start_node = 720575940613354467
visited = set()
queue = deque([start_node])
collected_ids = set()

sample_size = 200

logging.info(f"Beginning BFS subsample... sample_size={sample_size}")
while queue and len(collected_ids) < sample_size:
    curr = queue.popleft()
    if curr in visited:
        continue
    visited.add(curr)

    # Collect rows where pre_id == curr or post_id == curr
    matched_rows = df.index[(df["pre_pt_root_id"] == curr) | (df["post_pt_root_id"] == curr)]
    collected_ids.update(matched_rows)

    # Add neighbors to queue
    neighbors = adj.get(curr, [])
    for neighbor in neighbors:
        if neighbor not in visited:
            queue.append(neighbor)

    if len(collected_ids) >= sample_size:
        break

# Filter df
subset_df = df.loc[sorted(collected_ids)]
logging.info("Finished BFS subsample.")
# print(filtered_df)

def print_subset_breakdown(df: pd.DataFrame):
    breakdown = df.groupby('neuropil').agg(
        total_connections=('neuropil', 'count'),
        total_synapses=('syn_count', 'sum')
    ).reset_index().sort_values(by='total_synapses', ascending=False)

    print(breakdown.to_string(index=False))


logging.info(f"Calculating nueropil summary...")
print("Main dataset neuropil summary:")
print_subset_breakdown(df)
print(f"BFS subset (start_node={start_node}, sample_size={sample_size}) neuropil summary:")
print_subset_breakdown(subset_df)
logging.info(f"Finished calculating nueropil summaries.")

# Create a directed graph from the subset_df edges
logging.info("Creating directed graph from subset_df....")
G = nx.from_pandas_edgelist(
    subset_df,
    source="pre_pt_root_id",
    target="post_pt_root_id",
    edge_attr=True,  # Preserve edge attributes like syn_count, neuropil, etc.
    create_using=nx.DiGraph()
)
logging.info("Finished created directed graph from subset_df.")


logging.info("Creating inverted line graph from directed graph...")
L = nx.line_graph(G)
logging.info("Created inverted line graph from directed graph.")

print(L)

# edge_labels = {edge: f"{edge[0]}→{edge[1]}" for edge in L.nodes()}

# logging.info("Saving plot...")

# plot_name = f"inverted_subset_{start_node}_{sample_size}.png"

# plt.figure(figsize=(30, 30))  # Smaller fig size = faster render

# nx.draw_networkx_nodes(L, pos, node_size=5, node_color='black')
# nx.draw_networkx_edges(L, pos, arrows=False, width=0.2)

# plt.axis("off")
# plt.tight_layout()
# plt.savefig(plot_name, dpi=150)  # Lower DPI = faster
# plt.close()

# logging.info(f"Plot saved as {plot_name}.")

