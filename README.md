# GraphKit

Python script to analyze networks by computing centrality measures, determining communities and plotting graphs.

### Requirements

* **Python 3.6.8+**
* datashader>=0.10.0
* leidenalg>=0.8.3
* networkit>=7.0
* networkx>=2.3
* pandas>=0.25.3
* plotly>=3.10.0
* python-igraph>=0.8.3
* python-louvain>=0.14

### Usage

####

```
usage: graphkit [-h] [--output OUTPUT_NAME] [--seed SEED] [-c]
                [--attrs ATTRS [ATTRS ...]] [--normalized] [-g]
                [--delimiter DELIMITER] [--discard-trivial]
                [--edge-attrs EDGE_ATTRS [EDGE_ATTRS ...]] [--engine ENGINE]
                [--k-core K] [--max_nodes MAX_NODES]
                [--max-nodes-by-attr SORT_BY [SORT_BY ...]] [--no-self-loops]
                [--source-attr SOURCE_ATTR] [--target-attr TARGET_ATTR]
                [--undirected] [--unweighted] [-p] [--color COLORS]
                [--colorscale] [--dimensions {2,3}] [--groups GROUPS]
                [--iterations ITERATIONS] [--labels LABELS] [--layout LAYOUT]
                [--linlog] [--nohubs] [--positions POS]
                G

positional arguments:
  G                     Path to graph file

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT_NAME  Output path to write returned data
  --seed SEED           Specify random seed for predictable randomness
  -c, --compute         Set to compute attributes only
  --attrs ATTRS [ATTRS ...]
                        Available attributes: {'networkx': ['degree',
                        'in_degree', 'out_degree', 'bridging_centrality',
                        'bridging_coef', 'brokering_centrality'], 'networkit':
                        ['betweenness_centrality', 'betweenness_approx',
                        'betweenness_est', 'betweenness_kadabra',
                        'closeness_centrality', 'closeness_approx',
                        'clustering_centrality', 'eigenvector_centrality',
                        'katz_centrality', 'laplacian_centrality',
                        'page_rank', 'louvain'], 'igraph': ['leiden']}
  --normalized          Returns normalized (Min/Max) attribute values
  -g, --graph           Set to build graph only
  --delimiter DELIMITER
                        Specify file field delimiter (default: comma)
  --discard-trivial     Do not consider isolates (nodes without connections)
  --edge-attrs EDGE_ATTRS [EDGE_ATTRS ...]
                        Field names to consider as edge attributes
  --engine ENGINE       Pandas engine: 'c' (default), 'python' or 'python-fwf'
  --k-core K            K-coreness value to consider for nodes
  --max_nodes MAX_NODES
                        Maximum number of nodes to export or plot
  --max-nodes-by-attr SORT_BY [SORT_BY ...]
                        Attribute to sort nodes by
  --no-self-loops       Remove edges from a node to itself
  --source-attr SOURCE_ATTR
                        Field name to consider as source
  --target-attr TARGET_ATTR
                        Field name to consider as target
  --undirected          Set graph edges to directionless
  --unweighted          Set graph edges to weightless
  -p, --plot            Set to plot graph only
  --color COLORS        Set or get node colors from 1-dimensional series/file
                        (default: '#ccc')
  --colorscale          Enable color scale based on node sizes
  --dimensions {2,3}    Choose between 2-dimensional or 3-dimensional plot
                        (for 'kamada-kawai' only)
  --groups GROUPS       Get node groups from 1-dimensional series/file
                        dictionary
  --iterations ITERATIONS
                        Number of iterations to perform (default: 10; for
                        'forceatlas2' only)
  --labels LABELS       Get node labels from 1-dimensional series/file
                        dictionary
  --layout LAYOUT       Available layouts: 'circular', 'forceatlas2', 'kamada-
                        kawai', 'random'
  --linlog              Enable linlog mode for spatializing nodes (for
                        'forceatlas2' only)
  --nohubs              Enable nohubs mode for spatializing nodes (for
                        'forceatlas2' only)
  --positions POS       Get node positions from 2-dimensional or 3-dimensional
                        series/file dictionary
```

#### Interactive usage

A **[Jupyter notebook](notebook/notebook.ipynb)** is included with detailed instructions on execution.
