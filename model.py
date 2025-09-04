"""
Create a model of HIV in Zimbabwe
"""

# %% Imports and settings
import sciris as sc
import starsim as ss
import stisim as sti
import pandas as pd
import numpy as np


# %% Interventions
def get_testing_products():
    """
    Define HIV products and testing interventions
    """
    scaleup_years = np.arange(1990, 2021)  # Years for testing
    years = np.arange(1990, 2041)  # Years for simulation
    n_years = len(scaleup_years)
    fsw_prob = np.concatenate([np.linspace(0, 0.75, n_years), np.linspace(0.75, 0.85, len(years) - n_years)])
    low_cd4_prob = np.concatenate([np.linspace(0, 0.85, n_years), np.linspace(0.85, 0.95, len(years) - n_years)])
    gp_prob = np.concatenate([np.linspace(0, 0.5, n_years), np.linspace(0.5, 0.6, len(years) - n_years)])

    # FSW agents who haven't been diagnosed or treated yet
    def fsw_eligibility(sim):
        return sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art

    fsw_testing = sti.HIVTest(
        years=years,
        test_prob_data=fsw_prob,
        name='fsw_testing',
        eligibility=fsw_eligibility,
        label='fsw_testing',
    )

    # Non-FSW agents who haven't been diagnosed or treated yet
    def other_eligibility(sim):
        return ~sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art

    other_testing = sti.HIVTest(
        years=years,
        test_prob_data=gp_prob,
        name='other_testing',
        eligibility=other_eligibility,
        label='other_testing',
    )

    # Agents whose CD4 count is below 200.
    def low_cd4_eligibility(sim):
        return (sim.diseases.hiv.cd4 < 200) & ~sim.diseases.hiv.diagnosed

    low_cd4_testing = sti.HIVTest(
        years=years,
        test_prob_data=low_cd4_prob,
        name='low_cd4_testing',
        eligibility=low_cd4_eligibility,
        label='low_cd4_testing',
    )

    return fsw_testing, other_testing, low_cd4_testing


def make_hiv():
    """ Make HIV arguments for sim"""
    hiv = sti.HIV(
        beta_m2f=0.035,
        eff_condom=0.95,
        init_prev_data=pd.read_csv('data/init_prev_hiv.csv'),
        rel_init_prev=1.,
    )
    return hiv


def make_hiv_intvs():

    n_art = pd.read_csv(f'data/n_art.csv').set_index('year')
    n_vmmc = pd.read_csv(f'data/n_vmmc.csv').set_index('year')
    fsw_testing, other_testing, low_cd4_testing = get_testing_products()
    art = sti.ART(coverage_data=n_art)
    vmmc = sti.VMMC(coverage_data=n_vmmc)
    prep = sti.Prep()

    interventions = [
        fsw_testing,
        other_testing,
        low_cd4_testing,
        art,
        vmmc,
        prep,
    ]

    return interventions


def make_sim_pars(sim, calib_pars):
    """
    Update the simulation parameters with the calibration parameters
    """
    def set_par(sim=None, fullparname=None, new_val=None):
        modtype, module, parname = split_par(fullparname)
        if sim.initialized:
            sim[modtype][module].pars[parname] = new_val
        else:
            idx = [d.name for d in sim.pars[modtype]].index(module)
            sim.pars[modtype][idx].pars[parname] = new_val
        return

    def split_par(fullparname):
        """ Remove disease_ prefix """
        modname = fullparname.split('_')[0]
        if 'hiv' in fullparname:
            modtype = 'diseases'
        elif 'nw' in fullparname:
            modtype = 'networks'
        else:
            raise NotImplementedError(f'Parameter {fullparname} not recognized')
        parname = fullparname[fullparname.find('_')+1:]
        return modtype, modname, parname

    # Loop over the calibration parameters
    for fullparname, pars in calib_pars.items():

        if isinstance(pars, dict):
            v = pars['value']
        elif sc.isnumber(pars):
            v = pars
        else:
            raise NotImplementedError(f'Parameter {fullparname} not recognized')

        if fullparname in ['index', 'mismatch']:
            continue
        else:
            try:
                set_par(sim=sim, fullparname=fullparname, new_val=v)
            except:
                raise NotImplementedError(f'Parameter {fullparname} not recognized')

    return sim


def make_hiv_sim(start=1990, stop=2030, seed=1, use_calib=False, calib_folder=None, par_idx=0):
    """ Make the HIV sim """

    hiv = make_hiv()
    diseases = [hiv]
    intvs = make_hiv_intvs()

    sim_args = dict(start=start, stop=stop, rand_seed=seed, n_agents=5e3, use_migration=True, rel_death=0.8)
    sim = sti.Sim(
        **sim_args,  # Unpack the arguments for the sim
        diseases=diseases,
        demographics='zimbabwe',
        datafolder='data/',
        interventions=intvs,
    )

    # If using calibration parameters, update the simulation
    if use_calib:
        if calib_folder is None:
            calib_folder = 'results'
        pars_df = sc.loadobj(f'{calib_folder}/zim_hiv_pars.df')
        calib_pars = pars_df.iloc[par_idx].to_dict()
        sim.init()
        sim = make_sim_pars(sim, calib_pars)
        print(f'Using calibration parameters for scenario {scenario} and index {par_idx}')

    return sim



if __name__ == '__main__':

    # SETTINGS
    seed = 1
    do_run = True
    do_save = True

    if do_run:
        sim = make_hiv_sim(seed=seed, start=1990, stop=2030, use_calib=False)
        sim.run()
        df = sim.to_df(resample='year', use_years=True, sep='.')  # Use dots to separate columns
        if do_save: sc.saveobj(f'results/hiv_sim.df', df)

        # Process and plot
        from plot_sims import plot_hiv_sims
        df = sc.loadobj(f'results/hiv_sim.df')
        plot_hiv_sims(df, start_year=1990, end_year=2030, which='single')

    print('Done.')


