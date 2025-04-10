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
import random

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

def print_subset_breakdown(df: pd.DataFrame):
    # Fixed indentation
    breakdown = df.groupby('neuropil').agg(
        total_connections=('neuropil', 'count'),
        total_synapses=('syn_count', 'sum')
    ).reset_index().sort_values(by='total_synapses', ascending=False)

    print(breakdown.to_string(index=False))

def create_sampled_line_graph(sample_size=1000, seed=42):
    np.random.seed(seed)  # Added seed setting

    start_node = np.random.choice(df["pre_pt_root_id"].unique())  # Randomly choose start node
    visited = set()
    queue = deque([start_node])
    collected_ids = set()

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

    subset_df = df.loc[list(collected_ids)[:sample_size]]  # Ensure exact sample size

    G = nx.from_pandas_edgelist(
        subset_df,
        source="pre_pt_root_id",
        target="post_pt_root_id",
        edge_attr=True,  # Preserve edge attributes like syn_count, neuropil, etc.
        create_using=nx.DiGraph()
    )

    L = nx.line_graph(G)
    for edge in G.edges(data=True):
        node = (edge[0], edge[1])
        if node in L:
            L.nodes[node].update(edge[2])
        else:
            node = (edge[1], edge[0])
            if node in L:
                L.nodes[node].update(edge[2])

    return L


def random_neuropil(line_graph, _):
    neuropils = [line_graph.nodes[node].get('neuropil') for node in line_graph.nodes if line_graph.nodes[node].get('neuropil')]
    return random.choice(neuropils) if neuropils else None
    
def most_popular_neuropil(line_graph, _):
    neighbor_votes = {}
    
    for node in line_graph.nodes:
        neuropil = line_graph.nodes[node].get('neuropil')
        if neuropil:
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + 1
    
    if neighbor_votes:
        return max(neighbor_votes.items(), key=lambda x: x[1])[0]
    else:
        return None

def majority_vote_neuropil(line_graph, node):
    neighbor_votes = {}
    neighbors = list(line_graph.neighbors(node))
    
    for neighbor in neighbors:
        neuropil = line_graph.nodes[neighbor].get('neuropil')
        if neuropil:
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + 1
    
    predicted_neuropil = max(neighbor_votes.items(), key=lambda x: x[1])[0] if neighbor_votes else None
    return predicted_neuropil



def majority_vote_weighted_neuropil(line_graph, node):
    neighbor_votes = {}
    neighbors = list(line_graph.neighbors(node))
    
    for neighbor in neighbors:
        neuropil = line_graph.nodes[neighbor].get('neuropil')
        syn_count = line_graph.nodes[neighbor].get('syn_count', 0)
        if neuropil:
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + syn_count
    
    predicted_neuropil = max(neighbor_votes.items(), key=lambda x: x[1])[0] if neighbor_votes else None
    return predicted_neuropil

def log_weighted_majority_vote_neuropil(line_graph, node):
    neighbor_votes = {}
    neighbors = list(line_graph.neighbors(node))
    
    for neighbor in neighbors:
        neuropil = line_graph.nodes[neighbor].get('neuropil')
        syn_count = line_graph.nodes[neighbor].get('syn_count', 0)
        if neuropil and syn_count > 0:
            weight = np.log(syn_count)
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + weight
    
    predicted_neuropil = max(neighbor_votes.items(), key=lambda x: x[1])[0] if neighbor_votes else None
    return predicted_neuropil

def nt_type_weighted_majority_vote_neuropil(line_graph, node, k=5):
    neighbor_votes = {}
    neighbors = list(line_graph.neighbors(node))
    node_nt_type = line_graph.nodes[node].get('nt_type')
    
    for neighbor in neighbors:
        neuropil = line_graph.nodes[neighbor].get('neuropil')
        neighbor_nt_type = line_graph.nodes[neighbor].get('nt_type')
        
        if neuropil:
            vote_weight = k if (node_nt_type and neighbor_nt_type and node_nt_type == neighbor_nt_type) else 1
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + vote_weight
    
    predicted_neuropil = max(neighbor_votes.items(), key=lambda x: x[1])[0] if neighbor_votes else None
    return predicted_neuropil

def directional_majority_vote_neuropil(line_graph, node):
    neighbor_votes = {}
    neighbors = list(line_graph.neighbors(node))
    
    # Count incoming edges to the node and outgoing edges from its neighbors
    for neighbor in neighbors:
        neuropil = line_graph.nodes[neighbor].get('neuropil')
        if neuropil:
            # Check if the edge is incoming to the node
            if node in line_graph.predecessors(neighbor):
                neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + 1
            # Check if the edge is outgoing from the node
            elif neighbor in line_graph.successors(node):
                neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + 1

    # Calculate the product of incoming and outgoing counts
    product_votes = {neuropil: count * neighbor_votes.get(neuropil, 0) for neuropil, count in neighbor_votes.items()}
    
    # Determine the neuropil with the greatest product
    predicted_neuropil = max(product_votes.items(), key=lambda x: x[1])[0] if product_votes else None
    return predicted_neuropil

    
def trial(func, line_graph, node):
    if 'neuropil' not in line_graph.nodes[node]:
        return -1  # Node does not have a label

    actual_neuropil = line_graph.nodes[node]['neuropil']
    predicted_neuropil = func(line_graph, node)
    
    if predicted_neuropil is None:  # Handle case where prediction fails
        return 0
        
    if predicted_neuropil == actual_neuropil:
        return 1  # Correct prediction
    else:
        return 0  # Incorrect prediction

def validate(func, line_graph):
    total_trials = 0
    correct_predictions = 0
    
    for node in line_graph.nodes:
        result = trial(func, line_graph, node)
        if result != -1:  # Only count valid trials
            total_trials += 1
            correct_predictions += result
    
    accuracy = correct_predictions / total_trials if total_trials > 0 else 0
    return (total_trials, correct_predictions, accuracy)

print("Validating line graph predictions...")
for i in range(10):
    L = create_sampled_line_graph(sample_size=500, seed=i)
    print(f"----------------------- Seed: {i} -----------------------")
    print(f"Random neuropil                        : {validate(random_neuropil, L)}")
    print(f"Most popular neuropil                  : {validate(most_popular_neuropil, L)}")
    print(f"Majority vote neuropil                 : {validate(majority_vote_neuropil, L)}")
    print(f"Majority vote weighted neuropil        : {validate(majority_vote_weighted_neuropil, L)}")
    print(f"Log weighted majority vote neuropil    : {validate(log_weighted_majority_vote_neuropil, L)}")
    print(f"NT type weighted majority vote neuropil: {validate(nt_type_weighted_majority_vote_neuropil, L)}")
    print(f"Directional majority vote neuropil: {validate(directional_majority_vote_neuropil, L)}")
