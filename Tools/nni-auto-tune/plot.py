# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import matplotlib.pyplot as plt
import json
import os
import numpy as np
import argparse
import time
"""This is almost the same plot code as ann-benchmarks"""


def create_plot(path, x_scale, y_scale, results):
    qps, recalls, build_times, build_params, search_params = [], [], [], [], []
    data = []
    print(len(results))
    start = time.time()
    for i, p in enumerate(results):
        if i % 2000 == 0:
            print("processed", i, 'results in', round(time.time() - start),
                  'seconds')
            start = time.time()

        with open(os.path.join("results", path, p), "r") as f:
            try:
                res = json.load(f)
                data.append((res["qps"], res["recall"], res["build_time"],
                             res["build_params"], res["search_params"]))
            except:
                print(os.path.join("results", path, p))

    data.sort(key=lambda t: (-1 * t[0], -1 * t[1]))
    last_x = -1
    comparator = ((lambda xv, lx: xv > lx) if last_x < 0 else
                  (lambda xv, lx: xv < lx))
    with open(path + '.json', 'w') as f:
        for i, (q, r, b, bp, sp) in enumerate(data):
            if comparator(r, last_x):
                last_x = r
                recalls.append(r)
                qps.append(q)
                print("Selected point", i, "recalls:", r, "qps:", q,
                      "build time:", b, "with build params:", bp,
                      "and search params:", sp)
                f.write("Selected point" + str(i) + "recalls:" + str(r) +
                        "qps:" + str(q) + "build time:" + str(b) +
                        "with build params:" + str(bp) + "and search params:" +
                        str(sp) + "\n")

    handles, labels = [], []
    handle, = plt.plot(recalls,
                       qps,
                       '-',
                       label="SPTAG",
                       color="r",
                       ms=7,
                       mew=3,
                       lw=3,
                       linestyle='-',
                       marker='+')
    handles.append(handle)
    labels.append('sptag')

    ax = plt.gca()
    ax.set_ylabel("Queries per second (1/s)")
    ax.set_xlabel("Recall")

    if x_scale[0] == 'a':
        alpha = float(x_scale[1:])
        fun = lambda x: 1 - (1 - x)**(1 / alpha)
        inv_fun = lambda x: 1 - (1 - x)**alpha
        ax.set_xscale('function', functions=(fun, inv_fun))
        if alpha <= 3:
            ticks = [inv_fun(x) for x in np.arange(0, 1.2, .2)]
            plt.xticks(ticks)
        if alpha > 3:
            from matplotlib import ticker
            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(
        "Recall-Queries per second (1/s) tradeoff - up and to the right is better"
    )
    box = plt.gca().get_position()
    ax.legend(handles,
              labels,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              prop={'size': 9})
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.setp(ax.get_xminorticklabels(), visible=True)

    ax.spines['bottom']._adjust_location()

    plt.savefig(path + '.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path',
                        help='the path to load result from',
                        default='glove-100-angular')
    parser.add_argument(
        '-X',
        '--x_scale',
        help=
        'Scale to use when drawing the X-axis. Typically linear, logit or a2',
        default='a4')
    parser.add_argument('-Y',
                        '--y_scale',
                        help='Scale to use when drawing the Y-axis',
                        choices=["linear", "log", "symlog", "logit"],
                        default='log')
    args = parser.parse_args()

    if os.path.exists(os.path.join("results", args.path)):
        results = os.listdir(os.path.join("results", args.path))
    else:
        raise FileNotFoundError("No auto tune result found")
    print("ploting", args.path)
    create_plot(args.path, args.x_scale, args.y_scale, results)
