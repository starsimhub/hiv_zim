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
    def fsw_eligibility(mod):
        return mod.sim.networks.structuredsexual.fsw & ~mod.sim.diseases.hiv.diagnosed & ~mod.sim.diseases.hiv.on_art

    fsw_testing = sti.HIVTest(
        years=years,
        test_prob_data=fsw_prob,
        name='fsw_testing',
        eligibility=fsw_eligibility,
        label='fsw_testing',
    )

    # Non-FSW agents who haven't been diagnosed or treated yet
    def other_eligibility(mod):
        return ~mod.sim.networks.structuredsexual.fsw & ~mod.sim.diseases.hiv.diagnosed & ~mod.sim.diseases.hiv.on_art

    other_testing = sti.HIVTest(
        years=years,
        test_prob_data=gp_prob,
        name='other_testing',
        eligibility=other_eligibility,
        label='other_testing',
    )

    # Agents whose CD4 count is below 200.
    def low_cd4_eligibility(mod):
        return (mod.sim.diseases.hiv.cd4 < 200) & ~mod.sim.diseases.hiv.diagnosed

    low_cd4_testing = sti.HIVTest(
        years=years,
        test_prob_data=low_cd4_prob,
        name='low_cd4_testing',
        eligibility=low_cd4_eligibility,
        label='low_cd4_testing',
    )

    # ANC testing: test undiagnosed pregnant women in first trimester
    def anc_eligibility(mod):
        tri1 = mod.sim.demographics.pregnancy.tri1_uids
        return tri1[~mod.sim.diseases.hiv.diagnosed[tri1]]

    anc_testing = sti.HIVTest(
        test_prob_data=0.9,
        dt_scale=False,
        name='anc_testing',
        eligibility=anc_eligibility,
        label='anc_testing',
    )

    return fsw_testing, other_testing, low_cd4_testing, anc_testing


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
    fsw_testing, other_testing, low_cd4_testing, anc_testing = get_testing_products()
    art = sti.ART(coverage_data=n_art)
    vmmc = sti.VMMC(coverage_data=n_vmmc)
    prep = sti.Prep()

    interventions = [
        fsw_testing,
        other_testing,
        low_cd4_testing,
        anc_testing,
        art,
        vmmc,
        prep,
    ]

    return interventions


def make_sim_components(n_agents=5e3, start=1990, stop=2030, dt=1/12, verbose=1/12, seed=1):

    total_pop = {1970: 5.203e6, 1980: 7.05e6, 1985: 8.691e6, 1990: 9980999, 2000: 11.83e6}[start]
    sim_args = dict(total_pop=total_pop, start=start, stop=stop, dt=dt, verbose=verbose, rand_seed=seed)

    ####################################################################################################################
    # Demographic modules
    ####################################################################################################################
    fertility_data = pd.read_csv(f'data/asfr.csv')
    pregnancy = ss.Pregnancy(fertility_rate=fertility_data)
    death_data = pd.read_csv(f'data/deaths.csv')
    death = ss.Deaths(death_rate=death_data, rate_units=1)
    demographics = [pregnancy, death]

    ####################################################################################################################
    # People and networks
    ####################################################################################################################
    ppl = ss.People(n_agents, age_data=pd.read_csv(f'data/age_dist_{start}.csv', index_col='age')['value'])
    sexual = sti.FastStructuredSexual(
        prop_f0=0.8,
        prop_f2=0.05,
        prop_m0=0.65,
        f1_conc=0.05,
        f2_conc=0.25,
        m1_conc=0.15,
        m2_conc=0.3,
        p_pair_form=0.6,  # 0.6,
        condom_data=pd.read_csv(f'data/condom_use.csv'),
    )
    maternal = ss.MaternalNet(unit='month')
    networks = [sexual, maternal]

    ####################################################################################################################
    # Diseases
    ####################################################################################################################
    hiv = make_hiv()
    diseases = [hiv]

    ####################################################################################################################
    # Interventions and analyzers
    ####################################################################################################################
    intvs = make_hiv_intvs()

    return sim_args, demographics, ppl, networks, diseases, intvs


def make_hiv_sim(start=1990, stop=2030, seed=1):
    """ Make the HIV sim """
    sim_args, demographics, ppl, networks, diseases, intvs = make_sim_components(start=start, stop=stop, seed=seed)
    sim = ss.Sim(
        **sim_args,  # Unpack the arguments for the sim
        people=ppl,
        diseases=diseases,
        networks=networks,
        demographics=demographics,
        interventions=intvs,
        analyzers=[],
    )

    return sim


if __name__ == '__main__':

    # SETTINGS
    seed = 1
    do_run = True
    do_save = True

    if do_run:
        sim = make_hiv_sim(seed=seed, start=1990, stop=2030)
        sim.run()
        df = sim.to_df(resample='year', use_years=True, sep='.')  # Use dots to separate columns
        if do_save: sc.saveobj(f'results/hiv_sim.df', df)

        # Process and plot
        from plot_sims import plot_hiv_sims
        df = sc.loadobj(f'results/hiv_sim.df')
        plot_hiv_sims(df, start_year=1990, end_year=2030, which='single')

    print('Done.')


