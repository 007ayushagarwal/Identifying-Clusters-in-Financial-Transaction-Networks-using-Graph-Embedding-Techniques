# Identifying Clusters in Financial Transaction Networks using Graph Embedding Techniques

## Project Overview
This project investigates the application of various graph embedding techniques to identify meaningful clusters within a financial transaction network. We apply and compare three methods: **Node2Vec**, **Spectral Embedding**, and **Graph Convolutional Networks (GCNs)**. These techniques are used to transform graph structure data into low-dimensional vector representations for efficient clustering analysis.

## Dataset
The dataset consists of financial transactions between entities, where each transaction includes:
- **Sender**: The entity initiating the transaction.
- **Receiver**: The entity receiving the transaction.
- **Amount**: The monetary value of the transaction.

This dataset is represented as a graph where nodes correspond to entities, and directed edges correspond to the transactions between them.

## Algorithms Applied
### 1. Node2Vec
- **Purpose**: Learns continuous feature representations for nodes in the graph.
- **Process**:
  - Graph Construction
  - Embedding Extraction via Node2Vec
  - Clustering using K-Means
  - Visualization with t-SNE for dimensionality reduction

### 2. Spectral Embedding
- **Purpose**: Uses the eigenvectors of the graph Laplacian matrix to map nodes to a low-dimensional space.
- **Process**:
  - Graph Construction
  - Calculation of Laplacian Matrix
  - Eigenvalue Decomposition
  - Clustering using K-Means
  - Visualization with t-SNE

### 3. Graph Convolutional Networks (GCN)
- **Purpose**: Applies convolutional layers to the graph for learning node embeddings.
- **Process**:
  - Graph Construction
  - Embedding Extraction via GCN
  - Clustering using K-Means
  - Visualization with t-SNE

## Results
The results of each embedding technique were evaluated based on the quality of the identified clusters. The silhouette score was used to measure clustering performance, and t-SNE plots were generated to visualize the clusters in a 2D space.

- **Node2Vec**: Achieved optimal clustering with 3 clusters based on k-means and silhouette score analysis.
- **Spectral Embedding**: Detected 3 clusters with strong separation after eigenvalue analysis of the Laplacian matrix.
- **GCN**: Achieved optimal clustering with 3 clusters, supported by silhouette scores and t-SNE visualization.

## Dependencies
- Python 3.x
- NetworkX
- Node2Vec
- Scikit-learn
- TensorFlow or PyTorch (for GCN)
- Matplotlib for visualization
- t-SNE (via Scikit-learn)

## How to Run
1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Load the dataset (`payments.csv`) and ensure it is in the correct directory.
4. Run the Jupyter Notebook file to reproduce the results for Node2Vec, Spectral Embedding, and GCN methods.

