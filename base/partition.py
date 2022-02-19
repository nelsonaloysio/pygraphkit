import logging as log
from abc import abstractmethod

import leidenalg
import networkit as nk
import pandas as pd


class Partition():

    @abstractmethod
    def __init__(self):
        ''' Abstract method for "DIY" implementations. '''

    @staticmethod
    def louvain(nkG, to_undirected=True):
        '''
        Implementation of the Parallel Louvain method for community detection:
        https://doi.org/10.1088/1742-5468/2008/10/P10008
        '''
        nkGu = nk.graphtools.toUndirected(nkG) if to_undirected else nkG
        mod = nk.community.PLM(nkGu).run().getPartition()
        modules = mod.upperBound()
        modularity = nk.community.Modularity().getQuality(mod, nkGu)
        log.debug(
            f'Modules (Louvain): {modules} (m={modularity:.3f})')
        return pd.Series(
            pd.to_numeric(mod.getVector(), downcast='integer'),
            name='louvain_partition',
        )

    @staticmethod
    def leiden(iG):
        '''
        Implementation of the Leiden method for community detection:
        https://doi.org/10.1038/s41598-019-41695-z
        '''
        mod = leidenalg.find_partition(iG, leidenalg.ModularityVertexPartition)
        modules = max(mod.membership)+1 if mod.membership else 0
        modularity = mod.quality()
        log.debug(
            f'Modules (Leiden): {modules} (m={modularity:.3f})')
        return pd.Series(
            pd.to_numeric(mod.membership, downcast='integer'),
            name='leiden_partition',
        )