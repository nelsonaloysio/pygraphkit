from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Union

import networkx as nx
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as py
import plotly.express as px

pio.templates.default = 'none'

NODE_COLOR = '#ccc'
NODE_LINE_COLOR = '#000'
NODE_LINE_WIDTH = 1.0
NODE_OPACITY = 1.0
NODE_SIZE = 7

EDGE_LINE_COLOR = '#555'
EDGE_LINE_WIDTH = 1.0

ANNOTATION_FONT_COLOR = '#555'
ANNOTATION_FONT_SIZE = 12
ANNOTATION_OFFSET = 0.075

COLORSCALE = [
    "#4e9cd5",
    "#ffffff",
    "#f6e16d",
    "#f76717",
    "#d11e26",
]
COLORBAR_THICKNESS = 12
TITLE_FONT_SIZE = 16

FONT_COLOR = 'grey'
FONT_FAMILY = 'Raleway, Arial, sans-serif'
FONT_SIZE = 16
LEGEND_Y = 0.5

class Plot():

    @abstractmethod
    def __init__(self):
        ''' Abstract method for "DIY" implementations. '''

    def plot(
        self,
        G: nx.Graph,
        pos: Union[dict, pd.DataFrame],
        **kwargs,
    ) -> go.Figure:
        '''
        Returns Plotly figure from network graph.
        '''
        if isinstance(pos, dict):
            pos = pd.DataFrame.from_dict(
                pos,
                orient='index',
            )

        if pos.shape[1] == 2:
            return self._plot_2d(G, pos=pos, **kwargs)

        elif pos.shape[1] == 3:
            return self._plot_3d(G, pos=pos, **kwargs)

        raise ValueError(
            f"Plotly network graphs expect a number of >1, <=3 dimensions (got {pos.shape})."
        )

    @staticmethod
    def _plot_2d(
        G: nx.Graph,
        pos: Union[dict, pd.DataFrame],
        color: Union[str, dict] = NODE_COLOR,
        colorbar_title: bool = False,
        colorscale: list = COLORSCALE,
        groups: dict = None,
        labels: dict = None,
        max_labels: int = None,
        resizer: Callable[[float], float] = lambda x: NODE_SIZE+x*25,
        reversescale: bool = False,
        showarrow: bool = False,
        showlegend: bool = None,
        showscale: bool = False,
        showlabels: bool = False,
        size: Union[int, dict] = None,
        title: str = None,
        unlabeled: str = 'Nodes',
    ) -> go.Figure:
        '''
        Returns 2-dimensional Plotly figure from network graph.
        '''
        node_traces = []
        node_groups = defaultdict(list)

        if groups is None:
            groups = {}

        for node in G.nodes():
            group = groups.get(node, unlabeled)
            node_groups[group].append(node)

        if size is None:
            size = pd\
                .Series(dict(G.degree()))\
                .to_frame()\
                .apply(lambda x: (x / x.max()))[0]\
                .apply(resizer)\
                .dropna()\
                .to_dict()

        for group, nodes in node_groups.items():
            x_nodes, y_nodes = [], []

            for node in nodes:
                x, y = pos.loc[node]
                x_nodes += tuple([x])
                y_nodes += tuple([y])

            text_ = [labels.get(node, node) for node in nodes] if isinstance(labels, dict) else nodes
            size_ = [size.get(node, NODE_SIZE) for node in nodes] if isinstance(size, dict) else NODE_SIZE
            color_ = [color.get(node, NODE_COLOR) for node in nodes] if isinstance(color, dict) else color

            node_traces.append(
                go.Scatter(
                    x=x_nodes,
                    y=y_nodes,
                    mode='markers',
                    hoverinfo='text',
                    name=str(group).format(len(nodes)),
                    text=text_,
                    marker=dict(
                        color=color_ or size_ or NODE_COLOR,
                        colorscale=colorscale,
                        opacity=NODE_OPACITY,
                        reversescale=reversescale,
                        showscale=showscale,
                        size=size_,
                        colorbar=dict(
                            title=colorbar_title,
                            thickness=COLORBAR_THICKNESS,
                            titleside='bottom',
                            xanchor='left',
                        ),
                        line=dict(
                            color=NODE_LINE_COLOR,
                            width=NODE_LINE_WIDTH,
                        )
                    )
                )
            )

        x_edges, y_edges = [], []

        for edge in G.edges():
            x0, y0 = pos.loc[edge[0]]
            x1, y1 = pos.loc[edge[1]]
            x_edges += tuple([x0, x1, None])
            y_edges += tuple([y0, y1, None])

        edge_trace = go.Scatter(
            x=x_edges,
            y=y_edges,
            mode='lines',
            hoverinfo='none',
            line=dict(
                color=EDGE_LINE_COLOR,
                width=EDGE_LINE_WIDTH,
            ),
            name='Connections',
        )

        axis = dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        )

        fig = go.Figure(
            data=[edge_trace, *node_traces],
            layout=go.Layout(
                title=title,
                titlefont=dict(
                    size=TITLE_FONT_SIZE,
                ),
                showlegend=(showlegend if showlegend is not None else
                            True if len(node_traces) > 1 else False),
                legend=dict(
                    y=LEGEND_Y,
                    font=dict(
                        family=FONT_FAMILY,
                        size=FONT_SIZE,
                        color=FONT_COLOR,
                        ),
                    ),
                xaxis=axis,
                yaxis=axis,
            ),
        )

        if showlabels:
            fig['layout'].update(
                annotations=Plot._make_annotations(
                    nodes=list(G.nodes())[:max_labels],
                    pos=pos,
                    labels=labels,
                    showarrow=showarrow,
                ),
            )

        return fig

    @staticmethod
    def _plot_3d(
        G: nx.Graph,
        pos: Union[dict, pd.DataFrame],
        color: Union[str, dict] = NODE_COLOR,
        colorscale: list = COLORSCALE,
        labels: dict = None,
        max_labels: int = None,
        resizer: Callable[[float], float] = lambda x: NODE_SIZE+x*25,
        size: Union[int, dict] = None,
        title: str = None,
    ) -> go.Figure:
        '''
        Returns 3-dimensional Plotly figure from network graph.
        '''
        x_nodes, y_nodes, z_nodes = [], [], []
        x_edges, y_edges, z_edges = [], [], []

        for node in G.nodes():
            x, y, z = pos.loc[node]
            x_nodes += tuple([x])
            y_nodes += tuple([y])
            z_nodes += tuple([z])

        for edge in G.edges():
            x0, y0, z0 = pos.loc[edge[0]]
            x1, y1, z1 = pos.loc[edge[1]]
            x_edges += tuple([x0, x1, None])
            y_edges += tuple([y0, y1, None])
            z_edges += tuple([z0, z1, None])

        if size is None:
            size = pd\
                .Series(dict(G.degree()))\
                .to_frame()\
                .apply(lambda x: (x / x.max()))[0]\
                .apply(resizer)\
                .to_dict()

        text = [labels.get(node, node) for node in G.nodes()] if isinstance(labels, dict) else list(G.nodes())
        size = [size.get(node, NODE_SIZE) for node in G.nodes()] if isinstance(size, dict) else NODE_SIZE
        color = [color.get(node, NODE_COLOR) for node in G.nodes()] if isinstance(color, dict) else color

        node_trace = go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode='markers',
            hoverinfo='text',
            text=text,
            marker=dict(
                color=color or size or NODE_COLOR,
                colorscale=colorscale,
                opacity=NODE_OPACITY,
                size=size,
                symbol='circle',
                line=dict(
                    color=NODE_LINE_COLOR,
                    width=NODE_LINE_WIDTH,
                ),
            ),
        )

        edge_trace = go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode='lines',
            hoverinfo='none',
            line=dict(
                color=EDGE_LINE_COLOR,
                width=EDGE_LINE_WIDTH,
            ),
        )

        axis = dict(
            title='',
            showbackground=False,
            showgrid=False,
            showline=False,
            showspikes=False,
            showticklabels=False,
            zeroline=False,
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                width=1000,
                height=1000,
                showlegend=False,
                scene=dict(
                    xaxis=axis,
                    yaxis=axis,
                    zaxis=axis,
                ),
                margin=dict(
                    t=100,
                ),
                hovermode='closest',
                annotations=[
                    dict(
                        x=0,
                        y=0.1,
                        font=dict(
                            size=14,
                        ),
                        showarrow=False,
                        text='',
                        xanchor='left',
                        xref='paper',
                        yanchor='bottom',
                        yref='paper',
                    )
                ]
            )
        )

        return fig

    @staticmethod
    def _make_annotations(
        pos: pd.DataFrame,
        labels: dict = None,
        nodes: list = None,
        offset: Union[int, dict] = None,
        showarrow: bool = False,
    ) -> list:
        '''
        Adds node labels as text to Plotly 2-d figure.
        https://plot.ly/~empet/14683/networks-with-plotly/
        '''
        return [
            dict(
                text=labels.get(i, node) if labels else node,
                x=pos.loc[node][0],
                y=pos.loc[node][1] + (offset.get(node) if isinstance(offset, dict) else offset),
                xref='x1',
                yref='y1',
                font=dict(
                    color=ANNOTATION_FONT_COLOR,
                    size=ANNOTATION_FONT_SIZE,
                ),
                showarrow=showarrow,
            )
            for node in (nodes or pos.index)
        ]
