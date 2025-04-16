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

def create_sampled_graph(sample_size=1000, seed=42):
    np.random.seed(seed)  # Added seed setting

    start_node = np.random.choice(df["pre_pt_root_id"].unique())  # Randomly choose start node from pre_pt_root_id
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

    return G


def random_neuropil(graph, _):
    neuropils = {data['neuropil'] for _, _, data in graph.edges(data=True) if 'neuropil' in data}
    return random.choice(list(neuropils)) if neuropils else None

def most_popular_neuropil(graph, _):
    neighbor_votes = {}

    for _, _, data in graph.edges(data=True):
        neuropil = data.get('neuropil')
        if neuropil:
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + 1

    if neighbor_votes:
        return max(neighbor_votes.items(), key=lambda x: x[1])[0]
    else:
        return None

def majority_vote_neuropil(graph, edge):
    neighbor_votes = {}
    source, target = edge

    # Get all edges that share either the source or target vertex
    neighbor_edges = []
    for e in graph.edges(data=True):
        if source in e[:2] or target in e[:2]:
            if e[:2] != (source, target):  # Don't include the edge itself
                neighbor_edges.append(e)

    # Count votes for each neuropil from neighboring edges
    for _, _, data in neighbor_edges:
        neuropil = data.get('neuropil')
        if neuropil:
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + 1

    # Determine the neuropil with the most votes
    if neighbor_votes:
        predicted_neuropil = max(neighbor_votes, key=neighbor_votes.get)
        return predicted_neuropil
    else:
        return None

def majority_vote_weighted_neuropil(graph, edge):
    neighbor_votes = {}
    source, target = edge

    # Get all edges that share either the source or target vertex
    neighbor_edges = []
    for e in graph.edges(data=True):
        if source in e[:2] or target in e[:2]:
            if e[:2] != (source, target):  # Don't include the edge itself
                neighbor_edges.append(e)

    # Count weighted votes for each neuropil from neighboring edges
    for _, _, data in neighbor_edges:
        neuropil = data.get('neuropil')
        syn_count = data.get('syn_count', 0)  # Default to 0 if syn_count is missing
        if neuropil:
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + syn_count

    # Determine the neuropil with the most weighted votes
    if neighbor_votes:
        predicted_neuropil = max(neighbor_votes, key=neighbor_votes.get)
        return predicted_neuropil
    else:
        return None

def log_weighted_majority_vote_neuropil(graph, edge):
    neighbor_votes = {}
    source, target = edge

    # Get all edges that share either the source or target vertex
    neighbor_edges = []
    for e in graph.edges(data=True):
        if source in e[:2] or target in e[:2]:
            if e[:2] != (source, target):  # Don't include the edge itself
                neighbor_edges.append(e)

    # Count weighted votes for each neuropil from neighboring edges using logarithmic scaling
    for _, _, data in neighbor_edges:
        neuropil = data.get('neuropil')
        syn_count = data.get('syn_count', 0)  # Default to 0 if syn_count is missing
        if neuropil and syn_count > 0:  # Only consider positive syn_count for log scaling
            weight = np.log(syn_count)  # Apply logarithmic scaling
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + weight

    # Determine the neuropil with the most weighted votes
    if neighbor_votes:
        predicted_neuropil = max(neighbor_votes, key=neighbor_votes.get)
        return predicted_neuropil
    else:
        return None

def nt_type_weighted_majority_vote_neuropil(graph, edge, k=100):
    neighbor_votes = {}
    source, target = edge
    edge_nt_type = graph.edges[edge].get('nt_type')

    # Get all edges that share either the source or target vertex
    neighbor_edges = []
    for e in graph.edges(data=True):
        if source in e[:2] or target in e[:2]:
            if e[:2] != (source, target):  # Don't include the edge itself
                neighbor_edges.append(e)

    # Count votes with bonus for matching nt_type
    for _, _, data in neighbor_edges:
        neuropil = data.get('neuropil')
        neighbor_nt_type = data.get('nt_type')

        if neuropil:
            # Base vote is 1, multiply by k if nt_types match
            vote_weight = k if (edge_nt_type and neighbor_nt_type and edge_nt_type == neighbor_nt_type) else 1
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + vote_weight

    # Determine the neuropil with the most votes
    if neighbor_votes:
        predicted_neuropil = max(neighbor_votes, key=neighbor_votes.get)
        return predicted_neuropil
    else:
        return None

def directional_majority_vote_neuropil(graph, edge):
    u, v = edge
    incoming_votes = {}
    outgoing_votes = {}

    # Count incoming edges to u and outgoing edges from v for each neuropil
    for e in graph.edges(data=True):
        neuropil = e[2].get('neuropil')
        if neuropil:
            # Check if the edge is incoming to u
            if e[1] == u:
                incoming_votes[neuropil] = incoming_votes.get(neuropil, 0) + 1
            # Check if the edge is outgoing from v
            elif e[0] == v:
                outgoing_votes[neuropil] = outgoing_votes.get(neuropil, 0) + 1

    # Calculate the product of incoming and outgoing counts
    product_votes = {neuropil: incoming_votes.get(neuropil, 0) * outgoing_votes.get(neuropil, 0) for neuropil in incoming_votes.keys()}

    # Determine the neuropil with the greatest product
    if product_votes:
        predicted_neuropil = max(product_votes, key=product_votes.get)
        return predicted_neuropil
    else:
        return None


def normalized_majority_vote_neuropil(graph, edge):
    u, v = edge
    neighbor_votes = {}

    # Calculate total frequencies of each neuropil in the graph
    total_frequencies = {}
    for e in graph.edges(data=True):
        neuropil = e[2].get('neuropil')
        if neuropil:
            total_frequencies[neuropil] = total_frequencies.get(neuropil, 0) + 1

    # Get all edges that share either the source or target vertex
    neighbor_edges = []
    for e in graph.edges(data=True):
        if u in e[:2] or v in e[:2]:
            if e[:2] != (u, v):  # Don't include the edge itself
                neighbor_edges.append(e)

    # Count votes
    for _, _, data in neighbor_edges:
        neuropil = data.get('neuropil')
        if neuropil:
            neighbor_votes[neuropil] = neighbor_votes.get(neuropil, 0) + 1

    # Normalize votes by the log of total frequencies
    normalized_votes = {neuropil: vote / np.log(total_frequencies.get(neuropil, 1)) for neuropil, vote in neighbor_votes.items()}

    # Determine the neuropil with the most normalized votes
    if normalized_votes:
        predicted_neuropil = max(normalized_votes, key=normalized_votes.get)
        return predicted_neuropil
    else:
        return None



def jaccard_similarity_neuropil(graph, edge):
    u, v = edge

    # Get neighbors of u and v
    u_neighbors = set(graph.neighbors(u))
    v_neighbors = set(graph.neighbors(v))

    # Get all unique neuropils from neighboring edges
    neuropils = set()
    for e in graph.edges(data=True):
        if u in e[:2] or v in e[:2]:
            neuropil = e[2].get('neuropil')
            if neuropil:
                neuropils.add(neuropil)

    # Calculate Jaccard similarity for each neuropil
    best_similarity = -1
    predicted_neuropil = None

    for neuropil in neuropils:
        # Get neighbors that connect via edges with this neuropil
        u_neuropil_neighbors = {n for n in u_neighbors if graph.edges.get((u,n),{}).get('neuropil') == neuropil
                              or graph.edges.get((n,u),{}).get('neuropil') == neuropil}
        v_neuropil_neighbors = {n for n in v_neighbors if graph.edges.get((v,n),{}).get('neuropil') == neuropil
                              or graph.edges.get((n,v),{}).get('neuropil') == neuropil}

        # Calculate Jaccard similarity
        intersection = len(u_neuropil_neighbors & v_neuropil_neighbors)
        union = len(u_neuropil_neighbors | v_neuropil_neighbors)

        if union > 0:
            similarity = intersection / union
            if similarity > best_similarity:
                best_similarity = similarity
                predicted_neuropil = neuropil

    return predicted_neuropil

def adamic_adar(graph, edge):
    u, v = edge

    # Get neighbors of u and v
    u_neighbors = set(graph.neighbors(u))
    v_neighbors = set(graph.neighbors(v))

    # Get all unique neuropils from neighboring edges
    neuropils = set()
    for e in graph.edges(data=True):
        if u in e[:2] or v in e[:2]:
            neuropil = e[2].get('neuropil')
            if neuropil:
                neuropils.add(neuropil)

    # Calculate Adamic Adar similarity for each neuropil
    best_similarity = -1
    predicted_neuropil = None

    for neuropil in neuropils:
        # Get neighbors that connect via edges with this neuropil
        u_neuropil_neighbors = {n for n in u_neighbors if graph.edges.get((u,n),{}).get('neuropil') == neuropil
                              or graph.edges.get((n,u),{}).get('neuropil') == neuropil}
        v_neuropil_neighbors = {n for n in v_neighbors if graph.edges.get((v,n),{}).get('neuropil') == neuropil
                              or graph.edges.get((n,v),{}).get('neuropil') == neuropil}

        # Calculate Adamic Adar similarity
        intersection = u_neuropil_neighbors & v_neuropil_neighbors
        similarity = 0
        for z in intersection:
            z_neighbors = set(graph.neighbors(z))
            z_neuropil_neighbors = {n for n in z_neighbors if graph.edges.get((z,n),{}).get('neuropil') == neuropil
                              or graph.edges.get((n,z),{}).get('neuropil') == neuropil}
            if np.log(len(z_neuropil_neighbors)) != 0:
                similarity += 1.0/np.log(len(z_neuropil_neighbors))
        if similarity > best_similarity:
            best_similarity = similarity
            predicted_neuropil = neuropil

    return predicted_neuropil

def katz_unweighted_neuropil(graph, edge, beta=0.05, max_path_length=5):
    u, v = edge

    # Get all unique neuropils from neighboring edges
    neuropils = set()
    for e in graph.edges(data=True):
        if u in e[:2] or v in e[:2]:
            neuropil = e[2].get('neuropil')
            if neuropil:
                neuropils.add(neuropil)

    # Calculate Katz similarity for each neuropil
    best_similarity = -1
    predicted_neuropil = None

    for neuropil in neuropils:
        # Create a subgraph with only edges of this neuropil
        neuropil_edges = [(src, tgt) for src, tgt, data in graph.edges(data=True)
                         if data.get('neuropil') == neuropil]
        subgraph = nx.DiGraph()
        subgraph.add_edges_from(neuropil_edges)

        # Get adjacency matrix
        A = nx.adjacency_matrix(subgraph).toarray()
        nodes = list(subgraph.nodes())

        # Get node indices in the adjacency matrix
        try:
            i_u = nodes.index(u)
            i_v = nodes.index(v)
        except ValueError:
            # One or both nodes not in this neuropil's subgraph
            continue

        # Calculate powers of A for different path lengths
        katz_score = 0
        A_power = A.copy()  # Start with A^1

        for path_length in range(1, max_path_length + 1):
            if path_length > 1:
                A_power = A_power @ A  # Compute next power A^path_length

            # Add contribution of paths of this length
            katz_score += (beta ** path_length) * A_power[i_u, i_v]

        if katz_score > best_similarity:
            best_similarity = katz_score
            predicted_neuropil = neuropil

    return predicted_neuropil


def trial(func, graph, edge):
    # Check if the edge has a label
    if 'neuropil' not in graph.edges[edge]:
        return -1  # Edge does not have a label

    actual_neuropil = graph.edges[edge]['neuropil']
    predicted_neuropil = func(graph, edge)

    if predicted_neuropil is None:  # Handle case where prediction fails
        return 0

    if predicted_neuropil == actual_neuropil:
        return 1  # Correct prediction
    else:
        return 0  # Incorrect prediction

def validate(func, graph):
    total_trials = 0
    correct_predictions = 0

    for edge in graph.edges:
        result = trial(func, graph, edge)
        if result != -1:  # Only count valid trials
            total_trials += 1
            correct_predictions += result

    accuracy = correct_predictions / total_trials if total_trials > 0 else 0
    return (total_trials, correct_predictions, accuracy)

print("Validating...")
for i in range(10,20):
    G = create_sampled_graph(sample_size=1000, seed=i)
    print(f"----------------------- Seed: {i} -----------------------")
    print(f"Random neuropil                        : {validate(random_neuropil, G)}")
    print(f"Most popular neuropil                  : {validate(most_popular_neuropil, G)}")
    print(f"Majority vote neuropil                 : {validate(majority_vote_neuropil, G)}")
    print(f"Majority vote weighted neuropil      : {validate(majority_vote_weighted_neuropil, G)}")
    print(f"Log weighted majority vote neuropil  : {validate(log_weighted_majority_vote_neuropil, G)}")
    print(f"NT type weighted majority vote neuropil: {validate(nt_type_weighted_majority_vote_neuropil, G)}")
    print(f"Directional majority vote neuropil     : {validate(directional_majority_vote_neuropil, G)}")
    print(f"Normalized majority vote neuropil      : {validate(normalized_majority_vote_neuropil, G)}")
    print(f"Jaccard similarity neuropil          : {validate(jaccard_similarity_neuropil, G)}")
    print(f"Adamic Adar similarity neuropil          : {validate(adamic_adar, G)}")
    print(f"Katz unweighted neuropil (beta=0.05)     : {validate(katz_unweighted_neuropil, G)}")