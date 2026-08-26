from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from umap import UMAP

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "BioNeuralNet"
sys.path.insert(0, str(REPO))

from bioneuralnet.clustering.leiden import Leiden
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.network_embedding import GNNEmbedding
from bioneuralnet.utils import select_top_k_variance
from bioneuralnet.utils.graph import gen_similarity_graph


def adjacency_to_nx(adjacency: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    nodes = list(adjacency.index)
    graph.add_nodes_from(nodes)
    matrix = adjacency.to_numpy()
    rows, cols = np.nonzero(matrix)
    for i, j in zip(rows, cols):
        if i >= j:
            continue
        weight = float(matrix[i, j])
        if weight > 0:
            graph.add_edge(nodes[i], nodes[j], weight=weight)
    return graph


def filter_small_clusters(x: np.ndarray, labels: np.ndarray, minimum: int = 2):
    unique, counts = np.unique(labels, return_counts=True)
    retained = set(unique[counts >= minimum])
    mask = np.array([label in retained for label in labels])
    return x[mask], labels[mask]


def graph_modularity(graph: nx.Graph, labels: np.ndarray, nodes: list[str]) -> float:
    groups: dict[int, set[str]] = {}
    for node, label in zip(nodes, labels):
        groups.setdefault(int(label), set()).add(node)
    return nx.algorithms.community.modularity(
        graph, list(groups.values()), weight="weight"
    )


def average_correlations(expression: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    corr = np.corrcoef(expression)
    same, different = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            value = corr[i, j]
            if np.isfinite(value):
                (same if labels[i] == labels[j] else different).append(value)
    return float(np.mean(same)), float(np.mean(different))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def module_tests(
    data: pd.DataFrame,
    nodes: list[str],
    labels: np.ndarray,
    subtype: pd.Series,
) -> pd.DataFrame:
    rows = []
    for label in np.unique(labels):
        features = [node for node, assigned in zip(nodes, labels) if assigned == label]
        if len(features) < 5:
            continue
        matrix = data.loc[:, features].to_numpy()
        eigengene = PCA(n_components=1).fit_transform(
            matrix - matrix.mean(axis=0, keepdims=True)
        )[:, 0]
        groups = [eigengene[subtype.to_numpy() == value] for value in subtype.unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        rows.append(
            {"module": int(label), "size": len(features), "F": f_stat, "p": p_value}
        )
    result = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    result["q_BH"] = benjamini_hochberg(result["p"].to_numpy())
    return result


def main() -> None:
    np.random.seed(42)
    out = ROOT / "repro"
    out.mkdir(parents=True, exist_ok=True)

    brca = DatasetLoader("brca")
    rna = brca.data["rna"]
    methylation = brca.data["meth"]
    mirna = brca.data["mirna"]
    pam50 = brca.data["target"].iloc[:, 0]

    rna_selected = select_top_k_variance(rna, k=1000)
    methylation_selected = select_top_k_variance(methylation, k=1000)
    combined = pd.concat([rna_selected, methylation_selected, mirna], axis=1)

    feature_adjacency = gen_similarity_graph(
        combined, k=25, metric="cosine", mutual=False, self_loops=False
    )
    feature_graph = adjacency_to_nx(feature_adjacency)

    gnn = GNNEmbedding(
        adjacency_matrix=feature_adjacency,
        omics_data=combined,
        phenotype_data=pam50,
        phenotype_col=pam50.name,
        clinical_data=None,
        model_type="GAT",
        hidden_dim=64,
        layer_num=4,
        dropout=True,
        num_epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        gpu=False,
        seed=42,
        tune=False,
        output_dir=out / "gnn",
    )
    gnn.fit()
    embedding_frame = gnn.embed(as_df=True)
    nodes = list(feature_graph.nodes())
    embeddings = embedding_frame.loc[nodes].to_numpy()

    leiden = Leiden(
        G=feature_graph,
        embeddings=embeddings,
        use_edge_weights=True,
        resolution=0.25,
        random_state=42,
    )
    pure = leiden.run(refine_with_kmeans=False)
    hybrid = leiden.run(refine_with_kmeans=True, max_k_per_community=3)

    expression = combined.T.loc[nodes].to_numpy()
    pure_emb, pure_labels = filter_small_clusters(embeddings, pure)
    hybrid_emb, hybrid_labels = filter_small_clusters(embeddings, hybrid)
    pure_expr, pure_expr_labels = filter_small_clusters(expression, pure)
    hybrid_expr, hybrid_expr_labels = filter_small_clusters(expression, hybrid)
    within_pure, between_pure = average_correlations(expression, pure)
    within_hybrid, between_hybrid = average_correlations(expression, hybrid)

    module_metrics = {
        "feature_nodes": len(nodes),
        "feature_edges": feature_graph.number_of_edges(),
        "pure_clusters": int(np.unique(pure).size),
        "hybrid_clusters": int(np.unique(hybrid).size),
        "pure_sizes": sorted(np.unique(pure, return_counts=True)[1].tolist(), reverse=True),
        "hybrid_sizes": sorted(np.unique(hybrid, return_counts=True)[1].tolist(), reverse=True),
        "modularity_pure": graph_modularity(feature_graph, pure, nodes),
        "modularity_hybrid": graph_modularity(feature_graph, hybrid, nodes),
        "silhouette_embedding_pure": silhouette_score(pure_emb, pure_labels),
        "silhouette_embedding_hybrid": silhouette_score(hybrid_emb, hybrid_labels),
        "davies_bouldin_pure": davies_bouldin_score(pure_emb, pure_labels),
        "davies_bouldin_hybrid": davies_bouldin_score(hybrid_emb, hybrid_labels),
        "silhouette_expression_pure": silhouette_score(pure_expr, pure_expr_labels),
        "silhouette_expression_hybrid": silhouette_score(hybrid_expr, hybrid_expr_labels),
        "within_corr_pure": within_pure,
        "between_corr_pure": between_pure,
        "within_corr_hybrid": within_hybrid,
        "between_corr_hybrid": between_hybrid,
    }

    pure_tests = module_tests(combined, nodes, pure, pam50)
    hybrid_tests = module_tests(combined, nodes, hybrid, pam50)
    pure_tests.to_csv(out / "pure_module_tests.csv", index=False)
    hybrid_tests.to_csv(out / "hybrid_module_tests.csv", index=False)

    patient_adjacency = gen_similarity_graph(
        combined.T, k=15, metric="cosine", mutual=True, self_loops=False
    )
    patient_graph = nx.from_pandas_adjacency(patient_adjacency)
    patient_order = list(patient_graph.nodes())
    patient_embedding = PCA(n_components=16, random_state=42).fit_transform(
        combined.loc[patient_order].to_numpy()
    )
    truth = pam50.loc[patient_order].to_numpy()
    kmeans_labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(
        patient_embedding
    )
    patient_leiden = Leiden(
        G=patient_graph,
        embeddings=patient_embedding,
        use_edge_weights=True,
        resolution=1.0,
        random_state=42,
    )
    patient_pure = patient_leiden.run(refine_with_kmeans=False)
    patient_hybrid = patient_leiden.run(refine_with_kmeans=True, max_k_per_community=3)
    patient_metrics = {
        "patient_nodes": patient_graph.number_of_nodes(),
        "patient_edges": patient_graph.number_of_edges(),
        "kmeans_clusters": 5,
        "kmeans_ari": adjusted_rand_score(truth, kmeans_labels),
        "kmeans_nmi": normalized_mutual_info_score(truth, kmeans_labels),
        "leiden_clusters": int(np.unique(patient_pure).size),
        "leiden_ari": adjusted_rand_score(truth, patient_pure),
        "leiden_nmi": normalized_mutual_info_score(truth, patient_pure),
        "hybrid_clusters": int(np.unique(patient_hybrid).size),
        "hybrid_ari": adjusted_rand_score(truth, patient_hybrid),
        "hybrid_nmi": normalized_mutual_info_score(truth, patient_hybrid),
    }

    coords = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeddings)
    np.savez_compressed(
        out / "feature_results.npz",
        embeddings=embeddings,
        umap=coords,
        pure=pure,
        hybrid=hybrid,
        nodes=np.asarray(nodes),
    )
    with (out / "metrics.json").open("w") as handle:
        json.dump(
            {"module": module_metrics, "patient": patient_metrics},
            handle,
            indent=2,
        )
    print(json.dumps({"module": module_metrics, "patient": patient_metrics}, indent=2))


if __name__ == "__main__":
    main()
