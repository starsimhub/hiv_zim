"""
Plotting utils
"""
import sciris as sc
import starsim as ss
import numpy as np


def set_font(size=None, font='Libertinus Sans'):
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return

percentile_pairs = [[.01, .99], [.1, .9], [.25, .75]]  # Order by wide to narrow (for alpha shading in plots)
percentiles = [percentile for percentile_pair in percentile_pairs for percentile in percentile_pair]


def get_y(df, which, rname):
    if which == 'single': y = df[rname]
    elif which == 'multi': y = df[(rname, '50%')]
    return y

