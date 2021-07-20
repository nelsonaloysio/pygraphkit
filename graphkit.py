import logging as log
from typing import Union

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure

from .base import (
    Centrality,
    Graph,
    Layout,
    Partition,
    Plot,
    Subgraph
)

DEFAULT_LAYOUT = 'forceatlas2_layout'

NODE_ATTR = {
    'networkx': [
        'degree',
        'in_degree',
        'out_degree',
        'bridging_centrality',
        'bridging_coef',
        'brokering_centrality',
    ],
    'networkit': [
        'betweenness_centrality',
        'betweenness_approx',
        'betweenness_est',
        'betweenness_kadabra',
        'closeness_centrality',
        'closeness_approx',
        'clustering_centrality',
        'eigenvector_centrality',
        'katz_centrality',
        'laplacian_centrality',
        'page_rank',
        'louvain',
    ],
    'igraph': [
        'leiden',
    ],
}


class GraphKit(Graph, Centrality, Layout, Partition, Plot, Subgraph):

    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state

    def __call__(self,
        G: Union[nx.Graph, pd.DataFrame, list, str],
        attrs: list = None,
        max_nodes: int = None,
        sort_by: Union[str, list] = 'degree',
        **kwargs,
    ) -> dict:
        G = super().graph(G)

        if max_nodes is not None:
            log.info(f"Got {G.order()} nodes and "
                     f"{G.size()} edges (directed={G.is_directed()}).")

        if G.order() > (max_nodes or G.order()):
            sort_by = (
                self._compute(G, attrs=sort_by)
                    .sort_values(by=sort_by, ascending=False)
                    .index
                if isinstance(sort_by, str)
                else sort_by
            )
            G = self.subgraph(G, sort_by[:max_nodes])
            log.info(f"Graph limited to {G.order()} nodes and "
                     f"{G.size()} edges (sort_by='{sort_by}').")

        return self.compute(G, attrs=attrs, **kwargs)

    def compute(
        self,
        G: nx.Graph,
        attrs: list = ['degree'],
        normalized: bool = False,
        **kwargs,
    ) -> pd.DataFrame:

        df = pd.DataFrame()
        attrs = [attrs] if isinstance(attrs, str) else attrs
        valid_attrs = [x for x in NODE_ATTR.values() for x in x]

        for attr in attrs:
            if attr not in valid_attrs:
                raise RuntimeError(
                    f"Invalid node centrality attribute ('{attr}'). " +
                    f"Available choices: {valid_attrs}.")

        nkG = self.nx2nk(G) if any(
            attr in NODE_ATTR.get('networkit') for attr in attrs
        ) else None

        iG = self.nx2ig(G) if any(
            attr in NODE_ATTR.get('igraph') for attr in attrs
        ) else None

        for attr in attrs:
            df[attr] = getattr(self, attr)(
                nkG
                if attr in NODE_ATTR.get('networkit')
                else iG
                if attr in NODE_ATTR.get('igraph')
                else G
            ) if G.order() else ()

        if normalized:
            # All attributes except partitions (dtypes=int)
            cols = df.select_dtypes(float).columns
            df[cols] = df[cols].apply(
                lambda x: (x-x.min())/(x.max()-x.min())
            )

        df.index = G.nodes()
        df.index.name = 'id'
        return df

    def plot(
        self,
        G: nx.Graph,
        pos: Union[dict, pd.DataFrame] = None,
        discard_trivial: bool = True,
        layout: str = None,
        layout_opts: dict = None,
        max_nodes: int = None,
        sort_by: Union[str, list] = 'degree',
        **kwargs,
    ) -> Figure:
        order = G.order()

        if order > (max_nodes or G.order()):
            sort_by = (
                self._compute(G, attrs=sort_by)
                    .sort_values(by=sort_by, ascending=False)
                    .index
                if isinstance(sort_by, str)
                else sort_by
            )
            G = self.subgraph(G, sort_by[:max_nodes])

        if discard_trivial:
            # Remove nodes without connections
            G = self.subgraph(
                G, nodelist=(
                    self.nodes(G)
                        .drop(list(nx.isolates(G)))
                        .index
                )
            )

        if pos is None:
            layout_opts = {} if layout_opts is None else dict(layout_opts)
            if layout in ('forceatlas2_layout', 'random_layout'):
                layout_opts['seed'] = layout_opts.get('seed', self.random_state)
            pos = getattr(self, (layout or DEFAULT_LAYOUT))(G, **layout_opts)

        if G.order() != pos.shape[0]:
            G = self.subgraph(G,
                nodelist=pos.index
            )

        log.info(f"Generating graph plot (n={G.order()}, E={G.size()}).")
        return super().plot(G, pos=pos, **kwargs)
