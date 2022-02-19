#!/usr/bin/env python3

__git__ = 'git@github.com:nelsonaloiysio/GraphKit.git'
__url__ = 'https://github.com/nelsonaloysio/GraphKit'

from argparse import ArgumentParser
from os.path import isfile
from sys import argv

import pandas as pd

from graphkit import GraphKit
from graphkit import DEFAULT_LAYOUT, NODE_ATTR


def getargs(args=argv[1:]):
    parser = ArgumentParser(description=str('GraphKit | %s' % __url__))

    parser.add_argument("G",
                        action="store",
                        help="Path to graph file")

    parser.add_argument("--output",
                        default="output",
                        dest="output_name",
                        help=f"Output path to write returned data")

    parser.add_argument("--seed",
                        action="store",
                        help="Specify random seed for predictable randomness",
                        type=int)

    parser.add_argument("-c", "--compute",
                        action="store_const",
                        const="compute",
                        default="graph",
                        dest="method",
                        help="Set to compute attributes only")

    parser.add_argument("--attrs",
                        default=["in_degree","out_degree"],
                        help=f"Available attributes: {NODE_ATTR}",
                        nargs="+")

    parser.add_argument("--normalized",
                        action="store_true",
                        help="Returns normalized (Min/Max) attribute values")

    parser.add_argument("-g", "--graph",
                        action="store_const",
                        const="graph",
                        dest="method",
                        help="Set to build graph only")

    parser.add_argument("--delimiter",
                        help="Specify file field delimiter (default: comma)")

    parser.add_argument("--discard-trivial",
                        action="store_true",
                        help="Do not consider isolates (nodes without connections)")

    parser.add_argument("--edge-attrs",
                        action="append",
                        help="Field names to consider as edge attributes",
                        nargs="+")

    parser.add_argument("--engine",
                        default="c",
                        help="Pandas engine: 'c' (default), 'python' or 'python-fwf'")

    parser.add_argument("--k-core",
                        dest="k",
                        help=f"K-coreness value to consider for nodes",
                        type=int)

    parser.add_argument("--max_nodes",
                        action="store",
                        help="Maximum number of nodes to export or plot",
                        type=int)

    parser.add_argument("--max-nodes-by-attr",
                        help="Attribute to sort nodes by",
                        dest="sort_by",
                        nargs="+")

    parser.add_argument("--no-self-loops",
                        action="store_true",
                        help="Remove edges from a node to itself")

    parser.add_argument("--source-attr",
                        action="store",
                        help="Field name to consider as source")

    parser.add_argument("--target-attr",
                        action="store",
                        help="Field name to consider as target")

    parser.add_argument("--undirected",
                        action="store_false",
                        dest="directed",
                        help="Set graph edges to directionless")

    parser.add_argument("--unweighted",
                        action="store_false",
                        dest="weights",
                        help="Set graph edges to weightless")

    parser.add_argument("-p", "--plot",
                        action="store_const",
                        const="plot",
                        dest="method",
                        help="Set to plot graph only")

    parser.add_argument("--color",
                        action="store",
                        dest="colors",
                        help="Set or get node colors from 1-dimensional series/file (default: '#ccc')",
                        type=load)

    parser.add_argument("--colorscale",
                        action="store_true",
                        help="Enable color scale based on node sizes")

    parser.add_argument("--dimensions",
                        action="store",
                        choices=[2, 3],
                        default=2,
                        dest="dim",
                        help="Choose between 2-dimensional or 3-dimensional plot (for 'kamada-kawai' only)",
                        type=int)

    parser.add_argument("--groups",
                        action="store",
                        help="Get node groups from 1-dimensional series/file dictionary",
                        type=load)

    parser.add_argument("--iterations",
                        action="store",
                        default=10,
                        help="Number of iterations to perform (default: 10; for 'forceatlas2' only)",
                        type=int)

    parser.add_argument("--labels",
                        action="store",
                        help="Get node labels from 1-dimensional series/file dictionary",
                        type=load)

    parser.add_argument("--layout",
                        action="store",
                        help="Available layouts: 'circular', 'forceatlas2', 'kamada-kawai', 'random'",
                        type=lambda x: "%s_layout" % x.replace("-", "_"))

    parser.add_argument("--linlog",
                        action="store_true",
                        help="Enable linlog mode for spatializing nodes (for 'forceatlas2' only)")

    parser.add_argument("--nohubs",
                        action="store_true",
                        help="Enable nohubs mode for spatializing nodes (for 'forceatlas2' only)")

    parser.add_argument("--positions",
                        action="store",
                        dest="pos",
                        help="Get node positions from 2-dimensional or 3-dimensional series/file dictionary",
                        type=load)

    args = parser.parse_args(args)
    return vars(args)


def load(f, delimiter=None, engine=None) -> dict:
    return (
        pd.read_json(f)
            if isfile(f) and f.endswith(".json") else
        pd.read_csv(f, delimiter=delimiter, engine=engine, index_col=0, squeeze=True)
            if isfile(f) else
        f
    )


def main():
    args = getargs()
    gk = GraphKit(random_state=args.get("seed"))
    getattr(gk, args.pop("method"))(**args)


if __name__ == '__main__':
    main()