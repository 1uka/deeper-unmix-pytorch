import argparse
import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import museval

from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results")

    parser.add_argument("--evaldir", type=str,
                        help="Directory with .json evaluation results")
    parser.add_argument("--metrics", nargs="+",
                        default=["SDR", "SAR", "SIR"], type=str, help="Plot values for specific metrics")
    parser.add_argument("--targets", nargs="+",
                        default=["vocals"], type=str, help="Plot results for specific targets")

    args, _ = parser.parse_known_args()

    eval_path = Path(args.evaldir).expanduser()

    if not os.path.exists(eval_path):
        raise ValueError("--evaldir must be a valid path")

    methods = museval.MethodStore()
    methods.add_eval_dir(eval_path)
    methods.add_sisec18()

    oracles = [
        'IBM1', 'IBM2', 'IRM1', 'IRM2', 'MWF', 'IMSK'
    ]

    agg_df = methods.agg_frames_scores()

    df = methods.df.groupby(
        ["method", "track", "target", "metric"]
    ).median().reset_index()

    # Get sorting keys (sorted by median of SDR:vocals)
    df_sort_by = df[
        (df.metric == args.metrics[0]) &
        (df.target == args.targets[0])
    ]

    # sort methods by score
    methods_by_sdr = df_sort_by.score.groupby(
        df_sort_by.method
    ).median().sort_values().index.tolist()

    g = sns.FacetGrid(
        df, row="target", col="metric",
        row_order=args.targets,
        height=5, sharex=False, aspect=0.8
    )
    g = (g.map(
        sns.boxplot,
        "score",
        "method",
        orient='h',
        order=methods_by_sdr[::-1],
        showfliers=False,
        notch=True
    ).add_legend())

    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    g.fig.savefig(
        "boxplot.png",
        bbox_inches='tight',
    )
