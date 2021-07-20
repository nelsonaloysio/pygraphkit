import logging as log
from abc import abstractmethod

import networkx as nx
import pandas as pd


class Subgraph():

    @abstractmethod
    def __init__(self):
        ''' Abstract method for "DIY" implementations. '''

    @staticmethod
    def subgraph(G, nodelist=[]):
        ''' Returns graph object with nodes from a list. '''
        return nx.subgraph(G, nodelist).copy()

    @staticmethod
    def submodules(modules, submodules, max_modules=None, jacc_tol=0.5, best_match=False, weights=[]):
        ''' Returns dictionary of modules matching submodules. '''
        persistence = {}

        modules_ = modules.groupby(modules)
        submodules_ = submodules.groupby(submodules)

        list_modules = modules_.count().sort_values(ascending=False).index[:max_modules].tolist()
        list_submodules = submodules_.count().sort_values(ascending=False).index[:max_modules].tolist()

        for i in list_modules:
            for j in list_submodules:
                intersection = set(modules_.groups[i]).intersection(set(submodules_.groups[j]))
                union = set(modules_.groups[i]).union(set(submodules_.groups[j]))
                if len(weights):
                    union_weights = weights.loc[weights.index.intersection(union)].sum() # np.intersect1d(weights.index, union)
                    inter_weights = weights.loc[weights.index.intersection(intersection)].sum() # np.intersect1d(weights.index, intersection)
                else:
                    union_weights = len(union)
                    inter_weights = len(intersection)
                jacc = (inter_weights/union_weights) if union_weights else 0
                if jacc > jacc_tol:
                    if i not in persistence:
                        persistence[i] = {}
                    log.debug(f'Found module ({i}) submodule ({j}) (J={jacc:.3f}).')
                    persistence[i][j] = jacc

        log.debug(f'Found {len(persistence)} modules matching submodules (jacc_tol={jacc_tol}).')

        if best_match:
            return {
                x: sorted(
                    persistence[x].items(),
                    key=lambda x: x[1])[0][0]
                for x in persistence
            }

        return persistence

    @staticmethod
    def temporal_subgraphs(G, date_attr='timestamp', strftime='%Y-%m-%d', interval=1):
        ''' Returns dictionary of subgraphs over time. '''
        edgelist = nx.to_pandas_edgelist(G)

        if not G.order():
            return dict()

        if date_attr not in edgelist.columns:
            raise RuntimeError(f"Missing edge date attribute to determine subgraphs over time (date_attr='{date_attr}').")

        to_datetime = edgelist[date_attr]
        if not pd.api.types.is_datetime64_ns_dtype(to_datetime):
            try:
                to_datetime = pd.to_datetime(to_datetime, unit='s').copy()
            except ValueError:
                to_datetime = pd.to_datetime(to_datetime, infer_datetime_format=True).copy()

        dates = to_datetime.apply(lambda x: x.strftime(strftime))
        if not interval:
            n = 1  # a day
            length = len(dates.unique())
            if length >= 60:  # 2 months
                n = 30  # a month
            elif length >= 14:  # 2 weeks
                n = 7  # a week

        lst = sorted(list(dates.unique()))
        lst = [lst[i:i+n] for i in range(0, len(lst), n)]
        log.debug(f'Selected {len(lst)} buckets (interval={interval}d).')

        return {
            period[0]:
                nx.convert_matrix.from_pandas_edgelist(
                    edgelist.iloc[
                        to_datetime[
                            dates.apply(
                                lambda x: True if x in period else False
                            )
                        ].index],
                    source='source',
                    target='target',
                    edge_attr=edgelist.columns.tolist()[2:],
                    create_using=G,
                )
            for period in lst
        }