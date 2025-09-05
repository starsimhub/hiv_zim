"""
Run calibration for the HIV model
""" 
 
# Additions to handle numpy multithreading
import os
os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

# %% Imports and settings
import sciris as sc
import stisim as sti
import pandas as pd
from model import make_hiv_sim, make_sim_pars


# Run settings
debug = False  # If True, this will do smaller runs that can be run locally for debugging
n_trials = [1500, 2][debug]  # How many trials to run for calibration
n_workers = [50, 1][debug]    # How many cores to use
# storage = ["mysql://hpvsim_user@localhost/hpvsim_db", None][debug]  # Storage for calibrations
storage = None
do_shrink = True  # Whether to shrink the calibration results
make_stats = True  # Whether to make stats


def make_calibration(n_trials=None, n_workers=None):

    # Define the calibration parameters
    calib_pars = dict(
        # hiv_rel_init_prev=dict(low=1, high=5.0, guess=2.0),
        hiv_beta_m2f=dict(low=0.01, high=0.10, guess=0.05),
        nw_prop_f0 = dict(low=0.55, high=0.9, guess=0.85),
        nw_prop_m0 = dict(low=0.50, high=0.9, guess=0.81),
        nw_f1_conc = dict(low=0.01, high=0.2, guess=0.01),
        nw_m1_conc = dict(low=0.01, high=0.2, guess=0.01),
        nw_p_pair_form = dict(low=0.4, high=0.9, guess=0.5),
    )

    # Make the sim
    sim = make_hiv_sim(verbose=-1)
    data = pd.read_csv('data/zimbabwe_calib_data.csv')
    extra_results = ['hiv.n_diagnosed', 'hiv.n_on_art', 'n_alive']

    # Make the calibration
    calib = sti.Calibration(
        calib_pars=calib_pars,
        build_fn=make_sim_pars,
        sim=sim,
        extra_results=extra_results,
        data=data,
        total_trials=n_trials, n_workers=n_workers,
        die=True, reseed=False, storage=storage, save_results=True,
    )

    return sim, calib


def run_calibration(calib, n_trials=None, do_save=False):
    # Run the calibration
    printstr = f'Running calibration, {n_trials} trials'
    sc.heading(printstr)
    calib.calibrate()
    if do_save: sc.saveobj(f'results/zim_hiv_calib.obj', calib)
    return calib


if __name__ == '__main__':

    load_partial = False
    sim, calib = make_calibration(n_trials=n_trials, n_workers=n_workers)

    if load_partial:
        # Load a partially-run calibration study
        import optuna as op
        print(calib.run_args.study_name)
        study = op.load_study(storage=calib.run_args.storage, study_name=calib.run_args.study_name)
        # calib.run_args.continue_db = True
        # calib.calibrate()
        output = study.optimize(calib.run_trial, n_trials=2)
        calib.best_pars = sc.objdict(study.best_params)
        calib.parse_study(study)
        print('Best pars:', calib.best_pars)

        # Tidy up
        calib.calibrated = True
        if not calib.run_args.keep_db:
            calib.remove_db()

    else:
        calib = run_calibration(calib, n_trials=n_trials, do_save=False)

    print(f'... finished calibration!')
    print(f'Best pars are {calib.best_pars}')
    resfolder = 'results/'  #if not constrain else 'results/constrained'  # NB constrained not in repo

    # Save the results
    print('Shrinking and saving...')
    if do_shrink:
        sc.saveobj(f'{resfolder}/zim_hiv_calib_BIG.obj', calib)
        calib = calib.shrink(n_results=int(n_trials//10))  # Save 10% best results
        sc.saveobj(f'{resfolder}/zim_hiv_calib.obj', calib)
    else:
        sc.saveobj(f'{resfolder}/zim_hiv_calib.obj', calib)

    # Save the parameter dataframe
    sc.saveobj(f'{resfolder}/zim_hiv_pars.df', calib.df)

    if make_stats:
        print('Making stats...')
        from utils import percentiles
        df = calib.resdf
        df_stats = df.groupby(df.time).describe(percentiles=percentiles)
        sc.saveobj(f'{resfolder}/zim_hiv_calib_stats.df', df_stats)
        par_stats = calib.df.describe(percentiles=[0.05, 0.95])
        sc.saveobj(f'{resfolder}/zim_hiv_par_stats.df', par_stats)

    print('Done!')

