from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Union

import networkx as nx
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as py
import plotly.express as px

pio.templates.default = "none"

DEFAULT_NODE_COLOR = "#ccc"
DEFAULT_NODE_SIZE = 7

DEFAULT_COLORS = [
    "#006cb7",
    "#ff7700",
    "#00b035",
    "#ed0000",
    "#a643bd",
    "#965146",
    "#fb4cbe",
    "#7f7f7f",
    "#b2cb10",
    "#00c2d3",
]
DEFAULT_COLORSCALE = [
    "#4e9cd5",
    "#ffffff",
    "#f6e16d",
    "#f76717",
    "#d11e26",
]

DEFAULT_ANNOTATION_OFFSET = 0.075
DEFAULT_COLORBAR_THICKNESS = 12
DEFAULT_FONT_COLOR = "grey"
DEFAULT_FONT_FAMILY = "Raleway, Arial, sans-serif"
DEFAULT_FONT_SIZE = 16
DEFAULT_LEGEND_Y = 0.5
DEFAULT_TITLEFONT_SIZE = 16

class Plot():

    @abstractmethod
    def __init__(self):
        """ Abstract method for "DIY" implementations. """

    @staticmethod
    def plot(
        G: nx.Graph,
        pos: Union[dict, pd.DataFrame],
        colors: Union[str, dict] = DEFAULT_NODE_COLOR,
        colorbar_title: bool = False,
        colorbar_thickness: int = DEFAULT_COLORBAR_THICKNESS,
        colorscale: Union[list, bool] = None,
        edge_color: str = "#555",
        edge_width: float = 1.0,
        font_color: str = DEFAULT_FONT_COLOR,
        font_family: str = DEFAULT_FONT_FAMILY,
        font_size: int = DEFAULT_FONT_SIZE,
        groups: dict = None,
        height=1000,
        labels: dict = None,
        max_labels: int = None,
        node_line_color: str = "#000",
        node_line_width: float = 1.0,
        node_opacity: float = 1.0,
        resizer: Callable[[float], float] = lambda x: DEFAULT_NODE_SIZE+x*25,
        reversescale: bool = False,
        showarrow: bool = False,
        showbackground: bool = False,
        showgrid: bool = True,
        showlabels: bool = False,
        showlegend: bool = None,
        showline: bool = False,
        showscale: bool = False,
        showspikes: bool = False,
        showticklabels: bool = False,
        size: Union[int, dict] = None,
        title: str = None,
        titlefont_size: int = DEFAULT_TITLEFONT_SIZE,
        unlabeled: str = "Nodes",
        width=1000,
        zeroline: bool = False,
        **kwargs,
    ) -> go.Figure:

        if type(pos) == dict:
            pos = pd.DataFrame.from_dict(
                pos,
                orient="index",
            )

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

        for group in sorted(node_groups.keys()):
            x_nodes, y_nodes, z_nodes = [], [], []
            nodes = node_groups[group]

            for node in nodes:
                node_pos = pos.loc[node] # [x, y] or [x, y, z]
                x_nodes += tuple([node_pos[0]])
                y_nodes += tuple([node_pos[1]])
                if pos.shape[1] == 3:
                    z_nodes += tuple([node_pos[2]])

            text = [labels.get(node, node) for node in nodes] if type(labels) == dict else nodes
            size = [size.get(node, DEFAULT_NODE_SIZE) for node in nodes] if type(size) == dict else (size or DEFAULT_NODE_SIZE)
            color = [colors.get(node, DEFAULT_NODE_COLOR) for node in nodes] if type(colors) == dict else (colors or DEFAULT_NODE_COLOR)

            node_traces.append(
                (go.Scatter3d if pos.shape[1] == 3 else go.Scatter)(
                    x=x_nodes,
                    y=y_nodes,
                    mode="markers",
                    hoverinfo="text",
                    name=str(group).format(len(nodes)),
                    text=text,
                    marker=dict(
                        color=size if colorscale else color,
                        colorscale=colorscale if type(colorscale) == list else DEFAULT_COLORSCALE if colorscale == True else None,
                        opacity=node_opacity,
                        reversescale=reversescale,
                        showscale=showscale,
                        size=size,
                        colorbar=dict(
                            title=colorbar_title,
                            thickness=colorbar_thickness,
                            titleside="bottom",
                            xanchor="left",
                        ),
                        line=dict(
                            color=node_line_color,
                            width=node_line_width,
                        )
                    )
                )
            )
            if pos.shape[1] == 3:
                node_traces[-1].update(
                    dict(z=z_nodes)
                )

        x_edges, y_edges, z_edges = [], [], []

        for edge in G.edges():
            edge0_pos = pos.loc[edge[0]] # [x0, y0] or [x0, y0, z0]
            edge1_pos = pos.loc[edge[1]] # [x1, y1] or [x1, y1, z1]
            x_edges += tuple([edge0_pos[0], edge1_pos[0], None])
            y_edges += tuple([edge0_pos[1], edge1_pos[1], None])
            if pos.shape[1] == 3:
                z_edges += tuple([edge0_pos[2], edge1_pos[2], None])

        edge_trace = (go.Scatter3d if pos.shape[1] == 3 else go.Scatter)(
            x=x_edges,
            y=y_edges,
            mode="lines",
            hoverinfo="none",
            line=dict(
                color=edge_color,
                width=edge_width,
            ),
            name="Nodes",
        )
        if pos.shape[1] == 3:
            edge_trace.update(
                dict(z=z_edges)
            )

        axis = dict(
            showgrid=showgrid,
            showticklabels=showticklabels,
            zeroline=zeroline,
        )

        scene = dict(
            title="",
            showbackground=showbackground,
            showgrid=showgrid,
            showline=showline,
            showspikes=showspikes,
            showticklabels=showticklabels,
            zeroline=zeroline,
        )

        fig = go.Figure(
            data=[edge_trace, *node_traces],
            layout=go.Layout(
                height=height,
                legend=dict(
                    y=DEFAULT_LEGEND_Y,
                    font=dict(
                        family=font_family,
                        size=font_size,
                        color=font_color,
                        ),
                    ),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=showlegend if showlegend is not None else (True if len(node_traces) > 1 else False),
                title=title,
                titlefont=dict(
                    size=titlefont_size,
                ),
                width=width,
                xaxis=axis,
                yaxis=axis,
            ),
        )
        if pos.shape[1] == 3:
            fig.update_scenes(
                xaxis=scene,
                yaxis=scene,
                zaxis=scene,
            )

        if showlabels:
            fig.update_layout(
                annotations=Plot._make_annotations(
                    labels=labels,
                    nodes=list(G.nodes())[:max_labels],
                    pos=pos,
                    showarrow=showarrow,
                ),
            )

        return fig

    @staticmethod
    def _colors(colors=DEFAULT_COLORS):
        """
        Returns a sequence generator of discrete colors.

        See references below for built-in Plotly sequences:
            https://plotly.com/python/builtin-colorscales/
            https://plotly.com/python/colorscales/
            https://plotly.com/python/templates/
        """
        while 1:
            for clr in colors:
                yield clr

    @staticmethod
    def _make_annotations(
        pos: pd.DataFrame,
        color: str = "#555",
        labels: dict = None,
        nodes: list = None,
        offset: Union[int, dict] = DEFAULT_ANNOTATION_OFFSET,
        showarrow: bool = False,
        size: int = 12,
    ) -> list:
        """
        Adds node labels as text to Plotly 2-d figure.
        https://plot.ly/~empet/14683/networks-with-plotly/
        """
        return [
            dict(
                font=dict(
                    color=color,
                    size=size,
                ),
                showarrow=showarrow,
                text=labels.get(node) if labels else node,
                x=pos.loc[node][0],
                y=pos.loc[node][1] + (offset.get(node, DEFAULT_ANNOTATION_OFFSET) if type(offset) == dict else offset),
                xref="x1", # "paper",
                yref="y1", # "paper",
                # xanchor="left",
                # yanchor="bottom",
            )
            for node in (nodes or pos.index)
        ]
