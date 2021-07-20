import logging as log

import igraph as ig
import leidenalg
import networkx as nx
import networkit as nk
import pandas as pd

from abc import abstractmethod
from os.path import isfile, splitext
from typing import Union

READERS = {
    'gexf': nx.read_gexf,
    'gml': nx.read_gml,
    'graphml': nx.read_graphml,
    'leda': nx.read_leda,
    'pajek': nx.read_pajek,
    'pickle': nx.read_gpickle,
    'shp': nx.read_shp,
    'yaml': nx.read_yaml}

WRITERS = {
    'gexf': nx.write_gexf,
    'gml': nx.write_gml,
    'graphml': nx.write_graphml,
    'pajek': nx.write_pajek,
    'shp': nx.write_shp,
    'yaml': nx.write_yaml}


class Graph():

    @abstractmethod
    def __init__(self):
        ''' Abstract method for "DIY" implementations. '''

    def graph(self, data: Union[str, nx.Graph, nx.DiGraph, nk.Graph, ig.Graph, pd.DataFrame],
              source_attr=None, target_attr=None, edge_attr=None, directed=True):
        ''' Loads NetworkX graph from file or Pandas data frame. '''
        return data\
            if self.is_graph(
                data)\
            else self.nx_compose(
                data)\
            if isinstance(
                data,
                list)\
            else self.nx_read_graph(
                data)\
            if self.is_graph_file(
                data)\
            else self.pd2nx(
                self.pd_read_file(
                    data,
                ),
                source_attr=source_attr,
                target_attr=target_attr,
                edge_attr=edge_attr,
                create_using=nx.DiGraph if directed else nx.Graph)

    @staticmethod
    def nodes(G):
        ''' Returns Pandas node list from graph. '''
        return pd.DataFrame(dict(G.nodes(data=True)).values(), index=G.nodes())

    @staticmethod
    def edges(G):
        ''' Returns Pandas edge list from graph. '''
        return nx.to_pandas_edgelist(G)

    @staticmethod
    def adjacency(G):
        ''' Returns Pandas adjacency matrix from graph. '''
        return nx.to_pandas_adjacency(G)

    @staticmethod
    def density(G):
        ''' Returns graph density, measure of its completeness. '''
        return nx.density(G)

    @staticmethod
    def diameter(G):
        ''' Returns graph diameter, measure of its extension. '''
        return nx.diameter(G)

    @staticmethod
    def is_graph(instance, graphs=[nx.Graph, nx.DiGraph, nx.MultiGraph, nk.Graph, ig.Graph]):
        ''' Returns True if object is a known graph instance. '''
        return any(isinstance(instance, graph) for graph in graphs)

    @staticmethod
    def is_graph_file(filepath):
        ''' Returns True if NetworkX supports file format. '''
        return READERS.get(
            splitext(filepath)[1].lower().lstrip('.'),
            False)\
            if isinstance(
            filepath,
            str)\
            else False

    @staticmethod
    def nk2ig(nkG, index=[]):
        ''' Returns Networkit graph as igraph object. '''
        iG = ig.Graph(directed=nkG.isDirected())
        iG.add_vertices(list(nkG.iterNodes()) if not index else index)
        iG.add_edges(list(nkG.iterEdges()))
        iG.es['weight'] = list(nkG.iterEdgesWeights())
        return iG

    @staticmethod
    def nk2nx(nkG, index={}):
        ''' Returns Networkit graph as NetworkX object. '''
        G = nk.nxadapter.nk2nx(nkG)
        G = nx.relabel.relabel_nodes(G, index)
        return G

    @staticmethod
    def nx2ig(G):
        ''' Returns NetworkX graph as igraph object. '''
        iG = ig.Graph(directed=G.is_directed())
        iG.add_vertices(list(G.nodes()))
        iG.add_edges(list(G.edges()))
        edgelist = nx.to_pandas_edgelist(G)
        for attr in edgelist.columns[2:]:
            iG.es[attr] = edgelist[attr]
        return iG

    @staticmethod
    def nx2nk(G):
        ''' Returns NetworkX graph as Networkit object. '''
        return nk.nxadapter.nx2nk(G) if G.order() > 0 else nk.Graph()

    @staticmethod
    def nx_compose(list_of_graphs):
        ''' Returns a NetworkX graph composed from a list of graphs. '''
        C = list_of_graphs[0]
        for G in list_of_graphs[1:]:
            C = nx.compose(C, G)
        return C

    @staticmethod
    def nx_read_graph(path, ext=None):
        ''' Returns a NetworkX graph object from file, if supported. '''
        if isfile(path):
            if not ext:
                ext = splitext(path)[1].lower().lstrip('.')
            if READERS.get(ext):
                return READERS[ext](path)
            raise RuntimeError(f"Unidentified file extension (ext='{ext}').\n" +
                               f"Accepted formats: {list(READERS.keys())}.")
        raise FileNotFoundError(f"File '{path}' not found.")

    @staticmethod
    def nx_set_node_attrs(G, df: pd.DataFrame, attrs=[]):
        ''' Returns NetworkX Graph object with node attributes. '''
        for attr in df.columns:
            nx.set_node_attributes(G, df[attr], attr)
        return G

    @staticmethod
    def nx_write_graph(G, filepath, centrality=None, ext=None):
        ''' Writes a NetworkX graph object to file, if supported. '''
        if not ext:
            ext = splitext(filepath)[1].lower().lstrip('.')
        if isinstance(centrality, pd.DataFrame):
            for attr in centrality.columns:
                nx.set_node_attributes(G, centrality[attr].astype(str), attr)
        if WRITERS.get(ext):
            return WRITERS[ext](G, filepath)
        raise RuntimeError(f"Unidentified file extension (ext='{ext}').\n" +
                           f"Accepted formats: {list(WRITERS.keys())}.")

    @staticmethod
    def pd2nx(df: pd.DataFrame, source_attr=None, target_attr=None, edge_attr=[],
              create_using=nx.DiGraph, remove_self_loops=False, weighted=True):
        ''' Returns a NetworkX graph object from Pandas data frame. '''
        if not (source_attr and target_attr):
            if df.shape[1] == 2:
                source_attr, target_attr = df.columns.tolist()
            else:
                raise RuntimeError(
                    f"Missing 'source_attr' and 'target_attr' attributes for building graphs from Pandas.DataFrame objects.")
        # Check if data frame is empty
        if any(x == 0 for x in df.shape):
            log.warning(
                f"Received empty data frame {df.shape}, returning empty graph.")
            return nx.empty_graph(create_using=create_using)
        # Get edge list to convert to graph
        E = df[[source_attr, target_attr]]
        # Remove node self-connections
        if remove_self_loops:
            E = E[E[source_attr] != E[target_attr]]
        # Consider edge weights
        if weighted:
            weights = E.value_counts()
            edge_attr = ['weight'] + (edge_attr if edge_attr else [])
            E['weight'] = [weights.loc[x, y] for x, y in zip(E['source'], E['target'])]
        # Return graph with edge attributes
        return nx.convert_matrix\
                 .from_pandas_edgelist(
                     E,
                     source=source_attr,
                     target=target_attr,
                     edge_attr=edge_attr if edge_attr else None,
                     create_using=create_using)

    @staticmethod
    def pd_read_file(path_or_df: Union[str, pd.DataFrame], low_memory=False, sep=None, usecols=[]):
        ''' Returns a Pandas data frame object from file. '''
        def get_file_delimiter(path):
            ''' Returns character delimiter from file. '''
            delimiters = ['|', '\t', ';', ',']
            with open(path, 'rt') as f:
                header = f.readline()
            for char in delimiters:
                if char in header:
                    return char
            return '\n'

        return path_or_df\
            if isinstance(path_or_df, pd.DataFrame)\
            else pd.read_json(path_or_df)\
            if path_or_df.endswith('.json')\
            else pd.read_table(
                path_or_df,
                usecols=usecols if usecols else None,
                sep=sep or get_file_delimiter(path_or_df),
                low_memory=low_memory)