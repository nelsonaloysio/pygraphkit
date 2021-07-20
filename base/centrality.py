import logging as log
from abc import abstractmethod
from typing import Union

import networkit as nk
import networkx as nx
import pandas as pd


class Centrality():

    @abstractmethod
    def __init__(self):
        ''' Abstract method for "DIY" implementations. '''

    @staticmethod
    def degree(G):
        ''' Implementation of degree centrality. '''
        return pd.Series(
            [x[1] for x in G.degree()],
            name='degree'
        ).astype(float)

    @staticmethod
    def in_degree(G):
        ''' Implementation of in-degree centrality. '''
        return pd.Series(
            [x[1] for x in getattr(G, 'in_degree', G.degree)],
            name='in_degree'
        ).astype(float)

    @staticmethod
    def out_degree(G):
        ''' Implementation of out-degree centrality. '''
        return pd.Series(
            [x[1] for x in getattr(G, 'out_degree', G.degree)],
            name='out_degree'
        ).astype(float)

    @staticmethod
    def bridging_centrality(G, betweenness={}, bridging_coef={}):
        '''
        Implementation of bridging centrality:
        https://cse.buffalo.edu/tech-reports/2006-05.pdf
        '''
        if not betweenness:
            betweenness = nx.betweenness_centrality(G)
        if not bridging_coef:
            bridging_coef = Centrality.bridging_coef(G)
            bridging_coef.index = G.nodes()
        return pd.Series(
            [betweenness[node] * bridging_coef[node] for node in G.nodes()],
            name='bridging_centrality')

    @staticmethod
    def bridging_coef(G, degree={}):
        '''
        Implementation of bridging coefficient:
        https://cse.buffalo.edu/tech-reports/2006-05.pdf
        '''
        bc = {}
        if not degree:
            degree = nx.degree_centrality(G)
        for node in G.nodes():
            bc[node] = 0
            if degree[node] > 0:
                neighbors_degree = dict(
                    nx.degree(G, nx.neighbors(G, node))).values()
                sum_neigh_inv_deg = sum(
                    (1.0/d) for d in neighbors_degree)
                if sum_neigh_inv_deg > 0:
                    bc[node] = (1.0/degree[node]) / sum_neigh_inv_deg
        return pd.Series(
            bc.values(),
            name='bridging_coef')

    @staticmethod
    def brokering_centrality(G, degree={}, clustering={}):
        '''
        Implementation of brokering centrality:
        https://doi.org/10.1093/gbe/evq064
        '''
        if not degree:
            degree = nx.degree_centrality(G)
        if not clustering:
            clustering = nx.clustering(G)
        return pd.Series(
            [(1 - clustering[node]) * degree[node] for node in G.nodes()],
            name='brokering_centrality')

    @staticmethod
    def betweenness_centrality(nkG, normalized=False):
        '''
        Implementation of betweenness centrality:
        https://doi.org/10.1080/0022250X.2001.9990249
        '''
        return pd.Series(
            nk.centrality.Betweenness(nkG, normalized).run().scores(),
            name='betweenness_centrality')

    @staticmethod
    def betweenness_approx(nkG, epsilon=0.01, delta=0.1, universal_constant=1.0):
        '''
        Implementation of approximate betweenness centrality:
        https://doi.org/10.1145/2556195.2556224
        '''
        return pd.Series(
            nk.centrality.ApproxBetweenness(
                nkG, epsilon, delta, universal_constant).run().scores(),
            name='betweenness_approx')

    @staticmethod
    def betweenness_est(nkG, n_samples=100, normalized=False, parallel_flag=False):
        '''
        Implementation of estimated betweenness centrality:
        http://doi.org/10.1137/1.9781611972887.9
        '''
        return pd.Series(
            nk.centrality.EstimateBetweenness(
                nkG, n_samples, normalized, parallel_flag).run().scores(),
            name='betweenness_est')

    @staticmethod
    def betweenness_kadabra(nkG, err=0.05, delta=0.8, deterministic=False, k=0):
        '''
        Implementation of kadabra betweenness centrality:
        https://arxiv.org/abs/1903.09422
        '''
        return pd.Series(
            nk.centrality.KadabraBetweenness(
                nkG, err, delta, deterministic, k).run().scores(),
            name='betwenness_kadabra')

    @staticmethod
    def closeness_centrality(nkG, normalized=True, variant: Union['generalized', 'standard'] = 'generalized'):
        '''
        Implementation of closeness centrality:
        https://www.theses.fr/2015USPCD010.pdf
        '''
        if variant == 'generalized':
            variant = nk.centrality.ClosenessVariant.Generalized
        elif variant == 'standard':
            variant = nk.centrality.ClosenessVariant.Standard
        else:
            raise ValueError(
                f"Invalid closeness variant (variant='{variant}').\n\n" +
                f"Available choices: 'standard' or 'generalized' (default).")
        return pd.Series(
            nk.centrality.Closeness(
                nkG, normalized, variant).run().scores(),
            name='closeness_centrality')

    @staticmethod
    def closeness_approx(nkG, n_samples=100, normalized=True):
        '''
        Implementation of approximate closeness centrality:
        https://doi.org/10.1145/2660460.2660465
        '''
        return pd.Series(
            nk.centrality.ApproxCloseness(
                nkG, n_samples, normalized).run().scores(),
            name='closeness_approx')

    @staticmethod
    def clustering_centrality(nkG, remove_self_loops=True, to_undirected=True, turbo=False):
        '''
        Implementation of local clustering coefficient:
        https://doi.org/10.1137/1.9781611973198.1

        Turbo mode aimed at graphs with skewed, high degree distribution:
        https://dl.acm.org/citation.cfm?id=2790175
        '''
        nkGu = nk.graphtools.toUndirected(nkG) if to_undirected else nkG
        nkGu.removeSelfLoops() if remove_self_loops else None
        return pd.Series(
            nk.centrality.LocalClusteringCoefficient(
                nkGu, turbo=turbo).run().scores(),
            name='clustering')

    @staticmethod
    def eigenvector_centrality(nkG):
        '''
        Implementierung der Eigenvektor-Zentralität:
        https://doi.org/10.1007%2FBF01449896
        '''
        return pd.Series(
            nk.centrality.EigenvectorCentrality(nkG).run().scores(),
            name='eigenvector')

    @staticmethod
    def page_rank(nkG, cc: Union['l1', 'l2'] = 'l2', max_iterations=None):
        '''
        Implementation of PageRank, a variant of eigenvector centrality:
        http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf
        '''
        pr = nk.centrality.PageRank(nkG)
        if max_iterations:
            pr.maxIterations = max_iterations
        if cc == 'l1':
            pr.norm = nk.centrality.Norm.l1norm
        elif cc != 'l2':
            raise ValueError(
                f"Invalid convergence criterion (cc='{cc}').\n\n" +
                f"Available choices: 'l1' or 'l2' (default).")
        return pd.Series(
            pr.run().scores(),
            name='page_rank')

    @staticmethod
    def katz_centrality(nkG):
        '''
        Implementation of Katz centrality:
        https://doi.org/10.1007/BF02289026
        '''
        return pd.Series(
            nk.centrality.KatzCentrality(nkG).run().scores(),
            name='katz_centrality')

    @staticmethod
    def laplacian_centrality(nkG):
        '''
        Implementation of Laplacian centrality:
        https://doi.org/10.1016/j.ins.2011.12.027
        '''
        return pd.Series(
            nk.centrality.LaplacianCentrality(nkG).run().scores(),
            name='laplacian_centrality')
