"""
Plot calibrations
Running this file will print a table which shows the mean and 90% credible interval of the calibrated parameters.
"""

# Import packages
import sciris as sc
from plot_sims import plot_hiv_sims
from utils import percentile_pairs


# %% Run as a script
if __name__ == '__main__':

    # Load files - these should all be committed to the repository
    df_filename = f'results/zim_hiv_calib_stats.df'
    par_filename = f'results/zim_hiv_par_stats.df'
    df_stats = sc.loadobj(df_filename)
    par_stats = sc.loadobj(par_filename) 

    # Plot settings
    plot_kwargs = dict(
        start_year=1990,
        end_year=2025,
        which='multi',
        percentile_pairs=percentile_pairs,
        title=f'hiv_calib',
    )

    # Plot
    df_stats['timevec'] = df_stats.index
    plot_hiv_sims(df_stats, **plot_kwargs)

    # Print posterior
    pars = [p for p in par_stats.columns if p not in ['index', 'mismatch']]
    for p in pars:
        print(f'{p}: {par_stats[p]["mean"]:.3f} ({par_stats[p]["5%"]:.3f}–{par_stats[p]["95%"]:.3f})')

    print('Done.')
