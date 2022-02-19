import logging as log
from inspect import signature
from os.path import isfile, splitext
from typing import Union
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log.basicConfig(format=log_format, level=log.INFO)

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure

try:
    from .base import (
        Centrality,
        Graph,
        Layout,
        Partition,
        Plot,
        Subgraph
    )
except:
    from base import (
        Centrality,
        Graph,
        Layout,
        Partition,
        Plot,
        Subgraph
    )

DEFAULT_LAYOUT = 'random_layout'

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

    def __init__(self, random_state=None):
        self.random_state = random_state

    def compute(
        self,
        G: Union[str, nx.Graph],
        attrs: list = ['degree'],
        normalized: bool = False,
        output_name: str = None,
        **kwargs,
    ) -> pd.DataFrame:

        G = self.graph(G, **kwargs)

        df = pd.DataFrame()
        attrs = [attrs] if type(attrs) == str else attrs
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

        if output_name:
            df.to_json(output_name)\
            if output_name.endswith(".json") else\
            df.to_csv(output_name if output_name.endswith(".csv") else f"{output_name}.csv")

        return df

    def graph(
        self,
        G: Union[str, nx.Graph],
        delimiter: str = None,
        directed: bool = True,
        discard_trivial: bool = True,
        edge_attrs: list = None,
        engine: str = None,
        k: int = None,
        max_nodes: int = None,
        output_name: str = None,
        self_loops: bool = True,
        sort_by: Union[str, list] = 'degree',
        source_attr: str = None,
        target_attr: str = None,
        weights: bool = False,
        **kwargs,
    ) -> nx.Graph:

        G = super().graph(
            G,
            delimiter=delimiter,
            directed=directed,
            edge_attrs=edge_attrs,
            engine=engine,
            source_attr=source_attr,
            target_attr=target_attr,
            weights=weights,
        )

        if max_nodes is not None:
            log.info(f"Got {G.order()} nodes and "
                     f"{G.size()} edges (directed={G.is_directed()}).")

        if k or not self_loops:
            G.remove_edges_from(nx.selfloop_edges(G))

        if G.order() > (max_nodes or G.order()):
            sort_by = (
                self.compute(G, attrs=sort_by)
                    .sort_values(by=sort_by, ascending=False)
                    .index
                if type(sort_by) in (list, str)
                else sort_by
            )
            G = self.subgraph(G, sort_by[:max_nodes])
            log.info(f"Graph limited to {G.order()} nodes and "
                     f"{G.size()} edges (sort_by='{sort_by}').")

        if k is not None:
            G = nx.k_core(G, k)

        if discard_trivial:
            G = self.subgraph(
                G, nodelist=(
                    self.nodes(G)
                        .drop(list(nx.isolates(G)))
                        .index
                )
            )

        if output_name:
            self.nx_write_graph(G, output_name if splitext(output_name)[1] else f"{output_name}.gexf")

        return G

    def plot(
        self,
        G: Union[str, nx.Graph],
        color: Union[str, dict, pd.Series] = None,
        groups: Union[str, dict, pd.Series] = None,
        labels: Union[str, dict, pd.Series] = None,
        layout: str = None,
        output_name: str = None,
        pos: Union[str, list, dict, pd.DataFrame] = None,
        **kwargs,
    ) -> Figure:

        G = self.graph(G, **kwargs)
        kwargs["seed"] = self.random_state

        if pos is None or type(pos) in (str, list):
            layout_func = getattr(self, layout or DEFAULT_LAYOUT)
            layout_opts = {x: kwargs.get(x) for x in signature(layout_func).parameters if x in kwargs}
            pos = layout_func(G, **layout_opts)

        if G.order() != pos.shape[0]:
            G = self.subgraph(
                G,
                nodelist=pos.index
            )

        log.info(f"Generating graph plot (n={G.order()}, E={G.size()}).")

        fig = super().plot(
            G,
            color=color,
            groups=groups,
            labels=labels,
            pos=pos,
            **kwargs,
        )

        if output_name:
            fig.write_html(output_name)\
                if output_name.endswith(".html") else\
            fig.write_image(output_name)\
                if splitext(output_name)[1] else\
            fig.write_html(f"{output_name}.html")

        return fig

