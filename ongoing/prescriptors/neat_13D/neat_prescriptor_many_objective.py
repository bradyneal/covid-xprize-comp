from collections import defaultdict

import numpy as np
import pandas as pd
import os
from copy import deepcopy
import datetime
import neat

from ongoing.prescriptors.base import BasePrescriptor, PRED_CASES_COL, CASES_COL, NPI_COLUMNS, NPI_MAX_VALUES
import ongoing.prescriptors.base as base

path = '13d_5days'
num_checkpoint = 70

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_PREFIX = os.path.join(ROOT_DIR, 'neat-checkpoint-')
# CONFIG_FILE = os.path.join(ROOT_DIR, '{}/config-prescriptor-multiobjective'.format(path))
CONFIG_FILE = os.path.join(ROOT_DIR, '{}/config-prescriptor-{}'.format(path, num_checkpoint))
TMP_PRED_FILE_NAME = os.path.join(ROOT_DIR, 'tmp_predictions_for_prescriptions', 'preds.csv')
TMP_PRESCRIPTION_FILE = os.path.join(ROOT_DIR, 'tmp_prescription.csv')

# Number of days the prescriptors will look at in the past.
# Larger values here may make convergence slower, but give
# prescriptors more context. The number of inputs of each neat
# network will be NB_LOOKBACK_DAYS * (NPI_COLUMNS + 1) + NPI_COLUMNS.
# The '1' is for previous case data, and the final NPI_COLUMNS
# is for IP cost information.
NB_LOOKBACK_DAYS = 21

# Number of countries to use for training. Again, lower numbers
# here will make training faster, since there will be fewer
# input variables, but could potentially miss out on useful info.
NB_EVAL_COUNTRIES = 10

# Number of prescriptions to make per country.
# This can be set based on how many solutions in PRESCRIPTORS_FILE
# we want to run and on time constraints.
NB_PRESCRIPTIONS = 10

# Number of days to fix prescribed IPs before changing them.
# This could be a useful toggle for decision makers, who may not
# want to change policy every day. Increasing this value also
# can speed up the prescriptor, at the cost of potentially less
# interesting prescriptions.
ACTION_DURATION = 15

# Range of days the prescriptors will be evaluated on.
# To save time during training, this range may be significantly
# shorter than the maximum days a prescriptor can be evaluated on.
EVAL_START_DATE = '2020-08-01'
EVAL_END_DATE = '2020-08-02'

# Maximum number of generations to run (unlimited if None)
NB_GENERATIONS = 200

# Path to file containing neat prescriptors. Here we simply use a
# recent checkpoint of the population from train_prescriptor.py,
# but this is likely not the most complementary set of prescriptors.
# Many approaches can be taken to generate/collect more diverse sets.
# Note: this set can contain up to 10 prescriptors for evaluation.
PRESCRIPTORS_FILE = os.path.join(ROOT_DIR, '{}/neat-checkpoint-{}'.format(path, num_checkpoint))


def dominates(one, other):
    """Return true if each objective of *one* is not strictly worse than
    the corresponding objective of *other* and at least one objective is
    strictly better.
    """
    not_equal = False
    for self_wvalue, other_wvalue in zip(one, other):
        if self_wvalue > other_wvalue:
            not_equal = True
        elif self_wvalue < other_wvalue:
            return False
    return not_equal

def sortNondominatedNSGA2(pop_arr, k, first_front_only=False):
    """Sort the first *k* *individuals* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists), the first list includes
              nondominated individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in pop_arr:
        map_fit_ind[ind.fitness_mult].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1:]:
            if dominates(fit_i, fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif dominates(fit_j, fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(pop_arr), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts


def assignCrowdingDist(individuals):
    """Assign a crowding distance to each individual's fitness.
    It is done per front.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness_mult, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness_mult)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    # find max and min distance
    max_val = -float("inf")
    min_val = float("inf")
    flag_plus_inf = False
    flag_minus_inf = False
    for dist in distances:
        if dist != float("inf") and max_val < dist:
            max_val = dist
            pass
        if dist != -float("inf") and min_val > dist:
            min_val = dist
            pass
        if dist == float("inf"):
            flag_plus_inf = True
        elif dist == -float("inf"):
            flag_minus_inf = True
        pass

    # set values equal to inf to be max + 0.5
    # set values equal to -inf to be max - 0.5
    # and rescale the rest
    if flag_plus_inf:
        max_val += 0.5
    if flag_minus_inf:
        min_val -= 0.5
    for i in range(0, len(distances)):
        if distances[i] == float("inf"):
            distances[i] = 1.
        elif distances[i] == -float("inf"):
            distances[i] = 0.
        else:
            distances[i] = (distances[i] - min_val) / (max_val - min_val)
            pass
        pass

    for i, dist in enumerate(distances):
        individuals[i].crowding_dist = dist / 2
        pass
    pass


def get_best_n_points_no_curve(n, arr_input):
    arr_list = arr_input.tolist()
    arr_list.sort(reverse=True, key=lambda x: x[0])
    arr_list = np.array(arr_list)

    # 2. find arc length
    arc_len_arr = []
    for pos in range(0, len(arr_input) - 1):
        p1 = np.array([arr_list[pos][0], arr_list[pos][1]])
        p2 = np.array([arr_list[pos + 1][0], arr_list[pos + 1][1]])
        arc_len_arr.append(np.linalg.norm(p2 - p1))
    arc_len_arr = np.array(arc_len_arr)
    # distance delta
    d = sum(arc_len_arr) / (n-1)
    # cumul_sum of art length
    arc_len_arr_cum = np.cumsum(arc_len_arr)

    # 3. choose ref. points
    # positions of reference points
    points_pos = [0]
    j = 1
    for i in range(0, len(arc_len_arr_cum)):
        if arc_len_arr_cum[i] >= j * d:
            points_pos.append(i+1)
            j += 1
            if j == n-1:
                break
        pass
    points_pos.append(len(arr_list) - 1)

    chosen_points = []
    for ref_point_pos in points_pos:
        ref_point = arr_list[ref_point_pos]
        dist = np.linalg.norm((arr_input - ref_point), axis=1)
        pos = np.argmin(dist)
        chosen_points.append(pos)
        pass

    return chosen_points

class Neat(BasePrescriptor):
    def __init__(self, seed=base.SEED, eval_start_date=EVAL_START_DATE, eval_end_date=EVAL_END_DATE,
                 nb_eval_countries=NB_EVAL_COUNTRIES, nb_lookback_days=NB_LOOKBACK_DAYS, nb_prescriptions=NB_PRESCRIPTIONS, nb_generations=NB_GENERATIONS,
                 action_duration=ACTION_DURATION, config_file=CONFIG_FILE, prescriptors_file=PRESCRIPTORS_FILE, hist_df=None, verbose=True):

        super().__init__(seed=seed)
        self.eval_start_date = pd.to_datetime(eval_start_date, format='%Y-%m-%d')
        self.eval_end_date = pd.to_datetime(eval_end_date, format='%Y-%m-%d')
        self.nb_eval_countries = nb_eval_countries
        self.nb_lookback_days = nb_lookback_days
        self.nb_prescriptions = nb_prescriptions
        self.nb_generations = nb_generations
        self.action_duration = action_duration
        self.config_file = config_file
        self.prescriptors_file = prescriptors_file
        self.hist_df = hist_df
        self.verbose = verbose

    def fit(self, hist_df=None):
        if hist_df is not None:
            self.hist_df = hist_df
        # As a heuristic, use the top NB_EVAL_COUNTRIES w.r.t. ConfirmedCases
        # so far as the geos for evaluation.
        eval_geos = list(self.hist_df.groupby('GeoID').max()['ConfirmedCases'].sort_values(
                         ascending=False).head(self.nb_eval_countries).index)
        if self.verbose:
            print("Nets will be evaluated on the following geos:", eval_geos)

        # Pull out historical data for all geos
        past_cases = {}
        past_ips = {}
        for geo in eval_geos:
            geo_df = self.hist_df[self.hist_df['GeoID'] == geo]
            past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))
            past_ips[geo] = np.array(geo_df[NPI_COLUMNS])

        # Gather values for scaling network output
        ip_max_values_arr = np.array([NPI_MAX_VALUES[ip] for ip in NPI_COLUMNS])

        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        if self.verbose:
            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(show_species_detail=True))

            # Add statistics reporter to provide extra info about training progress.
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)

        # Add checkpointer to save population every generation and every 10 minutes.
        p.add_reporter(neat.Checkpointer(generation_interval=1,
                                         time_interval_seconds=600,
                                         filename_prefix=CHECKPOINTS_PREFIX))

        # Function that evaluates the fitness of each prescriptor model, equal costs
        def eval_genomes_multy_ones(genomes, config):
            # Every generation sample a different set of costs per geo,
            # so that over time solutions become robust to different costs.
            cost_df = base.generate_costs(self.hist_df, mode='equal')
            cost_df = base.add_geo_id(cost_df)

            geo_costs = {}
            for geo in eval_geos:
                costs = cost_df[cost_df['GeoID'] == geo]
                cost_arr = np.array(costs[NPI_COLUMNS])[0]
                geo_costs[geo] = cost_arr
            # Evaluate each individual
            for genome_id, genome in genomes:
                if genome.fitness is not None:
                    continue
                # Create net from genome
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                # Set up dictionary to keep track of prescription
                df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
                for ip_col in NPI_COLUMNS:
                    df_dict[ip_col] = []
                # Set initial data
                eval_past_cases = deepcopy(past_cases)
                eval_past_ips = deepcopy(past_ips)
                # Compute prescribed stringency incrementally
                stringency = np.zeros(config.genome_config.num_outputs)
                # Make prescriptions one day at a time, feeding resulting
                # predictions from the predictor back into the prescriptor.
                for date in pd.date_range(self.eval_start_date, self.eval_end_date):
                    date_str = date.strftime("%Y-%m-%d")
                    # Prescribe for each geo
                    for geo in eval_geos:
                        # Prepare input data. Here we use log to place cases
                        # on a reasonable scale; many other approaches are possible.
                        X_cases = np.log(eval_past_cases[geo][-self.nb_lookback_days:] + 1)
                        X_ips = eval_past_ips[geo][-self.nb_lookback_days:]
                        X_costs = geo_costs[geo]
                        X = np.concatenate([X_cases.flatten(),
                                            X_ips.flatten(),
                                            X_costs])
                        # Get prescription
                        prescribed_ips = net.activate(X)
                        # Map prescription to integer outputs
                        prescribed_ips = (prescribed_ips * ip_max_values_arr).round()
                        # Add it to prescription dictionary
                        country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
                        df_dict['CountryName'].append(country_name)
                        df_dict['RegionName'].append(region_name)
                        df_dict['Date'].append(date_str)
                        for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
                            df_dict[ip_col].append(prescribed_ip)
                        # Update stringency. This calculation could include division by
                        # the number of IPs and/or number of geos, but that would have
                        # no effect on the ordering of candidate solutions.
                        stringency += prescribed_ips
                    # Create dataframe from prescriptions.
                    pres_df = pd.DataFrame(df_dict)
                    # Make prediction given prescription for all countries
                    pred_df = self.get_predictions(self.eval_start_date.strftime("%Y-%m-%d"), date_str, pres_df)
                    # Update past data with new day of prescriptions and predictions
                    pres_df = base.add_geo_id(pres_df)
                    pred_df = base.add_geo_id(pred_df)
                    new_pres_df = pres_df[pres_df['Date'] == date_str]
                    new_pred_df = pred_df[pred_df['Date'] == date_str]
                    for geo in eval_geos:
                        geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                        geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                        # Append array of prescriptions
                        pres_arr = np.array([geo_pres[ip_col].values[0] for ip_col in NPI_COLUMNS]).reshape(1, -1)
                        eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])
                        # Append predicted cases
                        eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                         geo_pred[PRED_CASES_COL].values[0])
                # Compute fitness. There are many possibilities for computing fitness and ranking
                # candidates. Here we choose to minimize the product of ip stringency and predicted
                # cases. This product captures the area of the 2D objective space that dominates
                # the candidate. We minimize it by including a negation. To place the fitness on
                # a reasonable scale, we take means over all geos and days. Note that this fitness
                # function can lead directly to the degenerate solution of all ips 0, i.e.,
                # stringency zero. To achieve more interesting behavior, a different fitness
                # function may be required.
                new_cases = pred_df[PRED_CASES_COL].mean().mean()
                fitness_mult = list(-stringency)
                fitness_mult.append(-new_cases)
                genome.fitness_mult = tuple(fitness_mult)
                if self.verbose:
                    print('Evaluated Genome', genome_id)
                    print('New cases:', new_cases)
                    print('Stringency:', stringency)
                    print('Fitness:', genome.fitness_mult)

        # Function that evaluates the fitness of each prescriptor model, equal costs
        def eval_genomes_2d_ones(genomes, config):
            # Every generation sample a different set of costs per geo,
            # so that over time solutions become robust to different costs.
            cost_df = base.generate_costs(self.hist_df, mode='equal')
            cost_df = base.add_geo_id(cost_df)

            geo_costs = {}
            for geo in eval_geos:
                costs = cost_df[cost_df['GeoID'] == geo]
                cost_arr = np.array(costs[NPI_COLUMNS])[0]
                geo_costs[geo] = cost_arr
            # Evaluate each individual
            for genome_id, genome in genomes:
                # Create net from genome
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                # Set up dictionary to keep track of prescription
                df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
                for ip_col in NPI_COLUMNS:
                    df_dict[ip_col] = []
                # Set initial data
                eval_past_cases = deepcopy(past_cases)
                eval_past_ips = deepcopy(past_ips)
                # Compute prescribed stringency incrementally
                stringency = 0.
                # Make prescriptions one day at a time, feeding resulting
                # predictions from the predictor back into the prescriptor.
                for date in pd.date_range(self.eval_start_date, self.eval_end_date):
                    date_str = date.strftime("%Y-%m-%d")
                    # Prescribe for each geo
                    for geo in eval_geos:
                        # Prepare input data. Here we use log to place cases
                        # on a reasonable scale; many other approaches are possible.
                        X_cases = np.log(eval_past_cases[geo][-self.nb_lookback_days:] + 1)
                        X_ips = eval_past_ips[geo][-self.nb_lookback_days:]
                        X_costs = geo_costs[geo]
                        X = np.concatenate([X_cases.flatten(),
                                            X_ips.flatten(),
                                            X_costs])
                        # Get prescription
                        prescribed_ips = net.activate(X)
                        # Map prescription to integer outputs
                        prescribed_ips = (prescribed_ips * ip_max_values_arr).round()
                        # Add it to prescription dictionary
                        country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
                        df_dict['CountryName'].append(country_name)
                        df_dict['RegionName'].append(region_name)
                        df_dict['Date'].append(date_str)
                        for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
                            df_dict[ip_col].append(prescribed_ip)
                        # Update stringency. This calculation could include division by
                        # the number of IPs and/or number of geos, but that would have
                        # no effect on the ordering of candidate solutions.
                        stringency += np.sum(geo_costs[geo] * prescribed_ips)
                    # Create dataframe from prescriptions.
                    pres_df = pd.DataFrame(df_dict)
                    # Make prediction given prescription for all countries
                    pred_df = self.get_predictions(self.eval_start_date.strftime("%Y-%m-%d"), date_str, pres_df)
                    # Update past data with new day of prescriptions and predictions
                    pres_df = base.add_geo_id(pres_df)
                    pred_df = base.add_geo_id(pred_df)
                    new_pres_df = pres_df[pres_df['Date'] == date_str]
                    new_pred_df = pred_df[pred_df['Date'] == date_str]
                    for geo in eval_geos:
                        geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                        geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                        # Append array of prescriptions
                        pres_arr = np.array([geo_pres[ip_col].values[0] for ip_col in NPI_COLUMNS]).reshape(1, -1)
                        eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])
                        # Append predicted cases
                        eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                         geo_pred[PRED_CASES_COL].values[0])
                # Compute fitness. There are many possibilities for computing fitness and ranking
                # candidates. Here we choose to minimize the product of ip stringency and predicted
                # cases. This product captures the area of the 2D objective space that dominates
                # the candidate. We minimize it by including a negation. To place the fitness on
                # a reasonable scale, we take means over all geos and days. Note that this fitness
                # function can lead directly to the degenerate solution of all ips 0, i.e.,
                # stringency zero. To achieve more interesting behavior, a different fitness
                # function may be required.
                new_cases = pred_df[PRED_CASES_COL].mean().mean()
                genome.fitness_mult = (-new_cases, -stringency)
                if self.verbose:
                    print('Evaluated Genome', genome_id)
                    print('New cases:', new_cases)
                    print('Stringency:', stringency)
                    print('Fitness:', genome.fitness_mult)

        # Function that evaluates the fitness of each prescriptor model, random costs
        def eval_genomes_2d(genomes, config):
            # Every generation sample a different set of costs per geo,
            # so that over time solutions become robust to different costs.
            cost_df = base.generate_costs(self.hist_df, mode='random')
            cost_df = base.add_geo_id(cost_df)

            geo_costs = {}
            for geo in eval_geos:
                costs = cost_df[cost_df['GeoID'] == geo]
                cost_arr = np.array(costs[NPI_COLUMNS])[0]
                geo_costs[geo] = cost_arr
            # Evaluate each individual
            for genome_id, genome in genomes:
                # Create net from genome
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                # Set up dictionary to keep track of prescription
                df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
                for ip_col in NPI_COLUMNS:
                    df_dict[ip_col] = []
                # Set initial data
                eval_past_cases = deepcopy(past_cases)
                eval_past_ips = deepcopy(past_ips)
                # Compute prescribed stringency incrementally
                stringency = 0.
                # Make prescriptions one day at a time, feeding resulting
                # predictions from the predictor back into the prescriptor.
                for date in pd.date_range(self.eval_start_date, self.eval_end_date):
                    date_str = date.strftime("%Y-%m-%d")
                    # Prescribe for each geo
                    for geo in eval_geos:
                        # Prepare input data. Here we use log to place cases
                        # on a reasonable scale; many other approaches are possible.
                        X_cases = np.log(eval_past_cases[geo][-self.nb_lookback_days:] + 1)
                        X_ips = eval_past_ips[geo][-self.nb_lookback_days:]
                        X_costs = geo_costs[geo]
                        X = np.concatenate([X_cases.flatten(),
                                            X_ips.flatten(),
                                            X_costs])
                        # Get prescription
                        prescribed_ips = net.activate(X)
                        # Map prescription to integer outputs
                        prescribed_ips = (prescribed_ips * ip_max_values_arr).round()
                        # Add it to prescription dictionary
                        country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
                        df_dict['CountryName'].append(country_name)
                        df_dict['RegionName'].append(region_name)
                        df_dict['Date'].append(date_str)
                        for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
                            df_dict[ip_col].append(prescribed_ip)
                        # Update stringency. This calculation could include division by
                        # the number of IPs and/or number of geos, but that would have
                        # no effect on the ordering of candidate solutions.
                        stringency += np.sum(geo_costs[geo] * prescribed_ips)
                    # Create dataframe from prescriptions.
                    pres_df = pd.DataFrame(df_dict)
                    # Make prediction given prescription for all countries
                    pred_df = self.get_predictions(self.eval_start_date.strftime("%Y-%m-%d"), date_str, pres_df)
                    # Update past data with new day of prescriptions and predictions
                    pres_df = base.add_geo_id(pres_df)
                    pred_df = base.add_geo_id(pred_df)
                    new_pres_df = pres_df[pres_df['Date'] == date_str]
                    new_pred_df = pred_df[pred_df['Date'] == date_str]
                    for geo in eval_geos:
                        geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                        geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                        # Append array of prescriptions
                        pres_arr = np.array([geo_pres[ip_col].values[0] for ip_col in NPI_COLUMNS]).reshape(1, -1)
                        eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])
                        # Append predicted cases
                        eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                         geo_pred[PRED_CASES_COL].values[0])
                # Compute fitness. There are many possibilities for computing fitness and ranking
                # candidates. Here we choose to minimize the product of ip stringency and predicted
                # cases. This product captures the area of the 2D objective space that dominates
                # the candidate. We minimize it by including a negation. To place the fitness on
                # a reasonable scale, we take means over all geos and days. Note that this fitness
                # function can lead directly to the degenerate solution of all ips 0, i.e.,
                # stringency zero. To achieve more interesting behavior, a different fitness
                # function may be required.
                new_cases = pred_df[PRED_CASES_COL].mean().mean()
                genome.fitness_mult = (-new_cases,  -stringency)
                if self.verbose:
                    print('Evaluated Genome', genome_id)
                    print('New cases:', new_cases)
                    print('Stringency:', stringency)
                    print('Fitness:', genome.fitness_mult)

        # Function that evaluates the fitness of each prescriptor model
        def eval_genomes(genomes, config):
            # Every generation sample a different set of costs per geo,
            # so that over time solutions become robust to different costs.
            cost_df = base.generate_costs(self.hist_df, mode='random')
            cost_df = base.add_geo_id(cost_df)

            geo_costs = {}
            for geo in eval_geos:
                costs = cost_df[cost_df['GeoID'] == geo]
                cost_arr = np.array(costs[NPI_COLUMNS])[0]
                geo_costs[geo] = cost_arr
            # Evaluate each individual
            for genome_id, genome in genomes:
                # Create net from genome
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                # Set up dictionary to keep track of prescription
                df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
                for ip_col in NPI_COLUMNS:
                    df_dict[ip_col] = []
                # Set initial data
                eval_past_cases = deepcopy(past_cases)
                eval_past_ips = deepcopy(past_ips)
                # Compute prescribed stringency incrementally
                stringency = 0.
                # Make prescriptions one day at a time, feeding resulting
                # predictions from the predictor back into the prescriptor.
                for date in pd.date_range(self.eval_start_date, self.eval_end_date):
                    date_str = date.strftime("%Y-%m-%d")
                    # Prescribe for each geo
                    for geo in eval_geos:
                        # Prepare input data. Here we use log to place cases
                        # on a reasonable scale; many other approaches are possible.
                        X_cases = np.log(eval_past_cases[geo][-self.nb_lookback_days:] + 1)
                        X_ips = eval_past_ips[geo][-self.nb_lookback_days:]
                        X_costs = geo_costs[geo]
                        X = np.concatenate([X_cases.flatten(),
                                            X_ips.flatten(),
                                            X_costs])
                        # Get prescription
                        prescribed_ips = net.activate(X)
                        # Map prescription to integer outputs
                        prescribed_ips = (prescribed_ips * ip_max_values_arr).round()
                        # Add it to prescription dictionary
                        country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
                        df_dict['CountryName'].append(country_name)
                        df_dict['RegionName'].append(region_name)
                        df_dict['Date'].append(date_str)
                        for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
                            df_dict[ip_col].append(prescribed_ip)
                        # Update stringency. This calculation could include division by
                        # the number of IPs and/or number of geos, but that would have
                        # no effect on the ordering of candidate solutions.
                        stringency += np.sum(geo_costs[geo] * prescribed_ips)
                    # Create dataframe from prescriptions.
                    pres_df = pd.DataFrame(df_dict)
                    # Make prediction given prescription for all countries
                    pred_df = self.get_predictions(self.eval_start_date.strftime("%Y-%m-%d"), date_str, pres_df)
                    # Update past data with new day of prescriptions and predictions
                    pres_df = base.add_geo_id(pres_df)
                    pred_df = base.add_geo_id(pred_df)
                    new_pres_df = pres_df[pres_df['Date'] == date_str]
                    new_pred_df = pred_df[pred_df['Date'] == date_str]
                    for geo in eval_geos:
                        geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                        geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                        # Append array of prescriptions
                        pres_arr = np.array([geo_pres[ip_col].values[0] for ip_col in NPI_COLUMNS]).reshape(1,-1)
                        eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])
                        # Append predicted cases
                        eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                         geo_pred[PRED_CASES_COL].values[0])
                # Compute fitness. There are many possibilities for computing fitness and ranking
                # candidates. Here we choose to minimize the product of ip stringency and predicted
                # cases. This product captures the area of the 2D objective space that dominates
                # the candidate. We minimize it by including a negation. To place the fitness on
                # a reasonable scale, we take means over all geos and days. Note that this fitness
                # function can lead directly to the degenerate solution of all ips 0, i.e.,
                # stringency zero. To achieve more interesting behavior, a different fitness
                # function may be required.
                new_cases = pred_df[PRED_CASES_COL].mean().mean()
                genome.fitness = -(new_cases * stringency)
                if self.verbose:
                    print('Evaluated Genome', genome_id)
                    print('New cases:', new_cases)
                    print('Stringency:', stringency)
                    print('Fitness:', genome.fitness)

        # Run until a solution is found. Since a "solution" as defined in our config
        # would have 0 fitness, this will run indefinitely and require manual stopping,
        # unless evolution finds the solution that uses 0 for all ips. A different
        # value can be placed in the config for automatic stopping at other thresholds.
        # winner = p.run(eval_genomes, n=self.nb_generations)
        # winner = p.run_NSGA2(eval_genomes_2d, n=self.nb_generations, multi=True, algo='NSGA-2')
        # winner = p.run_NSGA2(eval_genomes_2d_ones, n=self.nb_generations, multi=True, algo='PO')
        # winner = p.run_NSGA2(eval_genomes_multy_ones, n=self.nb_generations, multi=True, algo='PO')
        # winner = p.run_NSGA2(eval_genomes_multy_ones, n=self.nb_generations, multi=True, algo='PO-repair')

        return

    def prescribe(self,
                  start_date_str,
                  end_date_str,
                  prior_ips_df,
                  cost_df):

        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        geos = prior_ips_df['GeoID'].unique()

        # Restrict it to dates before the start_date
        df = self.hist_df[self.hist_df['Date'] <= start_date]

        # Create past case data arrays for all geos
        past_cases = {}
        for geo in geos:
            geo_df = df[df['GeoID'] == geo]
            past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))

        # Create past ip data arrays for all geos
        past_ips = {}
        for geo in geos:
            geo_df = prior_ips_df[prior_ips_df['GeoID'] == geo]
            past_ips[geo] = np.array(geo_df[NPI_COLUMNS])

        # Fill in any missing case data before start_date
        # using predictor given past_ips_df.
        # Note that the following assumes that the df returned by prepare_historical_df()
        # has the same final date for all regions. This has been true so far, but relies
        # on it being true for the Oxford data csv loaded by prepare_historical_df().
        last_historical_data_date_str = df['Date'].max()
        last_historical_data_date = pd.to_datetime(last_historical_data_date_str,
                                                   format='%Y-%m-%d')
        if last_historical_data_date + pd.Timedelta(days=1) < start_date:
            if self.verbose:
                print("Filling in missing data...")
            missing_data_start_date = last_historical_data_date + pd.Timedelta(days=1)
            missing_data_start_date_str = datetime.datetime.strftime(missing_data_start_date,
                                                               format='%Y-%m-%d')
            missing_data_end_date = start_date - pd.Timedelta(days=1)
            missing_data_end_date_str = datetime.datetime.strftime(missing_data_end_date,
                                                               format='%Y-%m-%d')
            pred_df = self.get_predictions(missing_data_start_date_str,
                                           missing_data_end_date_str,
                                           prior_ips_df)
            pred_df = base.add_geo_id(pred_df)

            for geo in geos:
                geo_df = pred_df[pred_df['GeoID'] == geo].sort_values(by='Date')
                pred_cases_arr = np.array(geo_df[PRED_CASES_COL])
                past_cases[geo] = np.append(past_cases[geo], pred_cases_arr)
        elif self.verbose:
            print("No missing data.")

        # Gather values for scaling network output
        ip_max_values_arr = np.array([NPI_MAX_VALUES[ip] for ip in NPI_COLUMNS])

        # Load prescriptors
        checkpoint = neat.Checkpointer.restore_checkpoint(self.prescriptors_file)
        # read population and prepare data
        pop = checkpoint.population
        pop_arr = [pop[key] for key in pop]
        # is population 2-objective?
        print('\t\tProblem dimensionality {}'.format(len(pop_arr[0].fitness_mult)))
        print('\t\tpop_size = {}'.format(len(pop_arr)))

        if len(pop_arr[0].fitness_mult) < 13:
            print('Less then 13 objectives!')
            exit(-2)
            pass
        else:
            # create new attribute for every genome
            for el in pop_arr:
                el.fitness_manyD = deepcopy(el.fitness_mult)
                pass
            pass
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.config_file)

        # Load IP costs to condition prescriptions
        geo_costs = {}
        geo_presc = {}
        all_chosen_nets = set()
        for geo in geos:
            costs = cost_df[cost_df['GeoID'] == geo]
            cost_arr = np.array(costs[NPI_COLUMNS])[0]
            geo_costs[geo] = cost_arr

            # find best prescriptors for this geo
            for el in pop_arr:
                npi_arr = np.array(el.fitness_manyD[0:-1])
                new_cases = el.fitness_manyD[-1]
                el.fitness_mult = (new_cases, sum(npi_arr*cost_arr))
                pass
            fronts = sortNondominatedNSGA2(pop_arr, len(pop))

            tmp_chosen = []
            for i in range(0, len(fronts)):
                tmp_chosen.extend(fronts[i])
                if len(tmp_chosen) >= NB_PRESCRIPTIONS:
                    break
                pass
            # print('!!!!!!!!!!! Choosing among {} prescriptors for geo {}'.format(len(tmp_chosen), geo))

            x_arr = [el.fitness_mult[0] for el in tmp_chosen]
            y_arr = [el.fitness_mult[1] for el in tmp_chosen]
            arr = np.array([x_arr, y_arr]).T
            chosen_points_pos = get_best_n_points_no_curve(NB_PRESCRIPTIONS, arr)
            chosen_prescriptors_ids = []
            for pos in chosen_points_pos:
                key_val = tmp_chosen[pos].key
                chosen_prescriptors_ids.append(key_val)
                all_chosen_nets.add(key_val)
            geo_presc[geo] = chosen_prescriptors_ids
            pass
        all_chosen_nets = list(all_chosen_nets)
        print('\t\tTotally {} nets were chosen'.format(len(all_chosen_nets)))

        # now create a dic of all required nets
        for key_val in all_chosen_nets:
            net = neat.nn.FeedForwardNetwork.create(pop[key_val], config)
            pop[key_val] = net
            pass

        print('\t\tGenerated {} nets'.format(len(all_chosen_nets)))

        # Generate prescriptions
        prescription_dfs = []
        for prescription_idx in range(0, NB_PRESCRIPTIONS):
            if self.verbose:
                print("Generating prescription", prescription_idx, "...")

            # Set up dictionary for keeping track of prescription
            df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
            for ip_col in sorted(NPI_MAX_VALUES.keys()):
                df_dict[ip_col] = []

            # Set initial data
            eval_past_cases = deepcopy(past_cases)
            eval_past_ips = deepcopy(past_ips)

            # Generate prescriptions iteratively, feeding resulting
            # predictions from the predictor back into the prescriptor.
            action_start_date = start_date
            while action_start_date <= end_date:

                # Get prescription for all regions
                for geo in geos:
                    # Prepare input data. Here we use log to place cases
                    # on a reasonable scale; many other approaches are possible.
                    X_cases = np.log(eval_past_cases[geo][-self.nb_lookback_days:] + 1)
                    X_ips = eval_past_ips[geo][-self.nb_lookback_days:]
                    X_costs = geo_costs[geo]
                    X = np.concatenate([X_cases.flatten(),
                                        X_ips.flatten(),
                                        X_costs])

                    # get current prescriptor
                    chosen_idx = geo_presc[geo][prescription_idx]
                    #prescriptor = pop[chosen_idx]
                    #print('\t\t{}: chosen {}'.format(geo, chosen_idx))
                    # Create net from genome
                    # net = neat.nn.FeedForwardNetwork.create(prescriptor, config)
                    net = pop[chosen_idx]
                    # Get prescription
                    prescribed_ips = net.activate(X)

                    # Map prescription to integer outputs
                    prescribed_ips = (prescribed_ips * ip_max_values_arr).round()

                    # Add it to prescription dictionary for the full ACTION_DURATION
                    country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
                    if region_name == 'nan':
                        region_name = np.nan
                    for date in pd.date_range(action_start_date, periods=self.action_duration):
                        if date > end_date:
                            break
                        date_str = date.strftime("%Y-%m-%d")
                        df_dict['CountryName'].append(country_name)
                        df_dict['RegionName'].append(region_name)
                        df_dict['Date'].append(date_str)
                        for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
                            df_dict[ip_col].append(prescribed_ip)

                # Create dataframe from prescriptions
                pres_df = pd.DataFrame(df_dict)

                # Make prediction given prescription for all countries
                pred_df = self.get_predictions(start_date_str, date_str, pres_df)

                # Update past data with new days of prescriptions and predictions
                pres_df = base.add_geo_id(pres_df)
                pred_df = base.add_geo_id(pred_df)
                for date in pd.date_range(action_start_date, periods=self.action_duration):
                    if date > end_date:
                        break
                    date_str = date.strftime("%Y-%m-%d")
                    new_pres_df = pres_df[pres_df['Date'] == date_str]
                    new_pred_df = pred_df[pred_df['Date'] == date_str]
                    for geo in geos:
                        geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                        geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                        # Append array of prescriptions
                        pres_arr = np.array([geo_pres[ip_col].values[0] for
                                             ip_col in NPI_COLUMNS]).reshape(1,-1)
                        eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

                        # It is possible that the predictor does not return values for some regions.
                        # To make sure we generate full prescriptions, this script continues anyway.
                        # This should not happen, but is included here for robustness.
                        if len(geo_pred) != 0:
                            eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                             geo_pred[PRED_CASES_COL].values[0])

                # Move on to next action date
                action_start_date += pd.DateOffset(days=self.action_duration)

            # Add prescription df to list of all prescriptions for this submission
            pres_df['PrescriptionIndex'] = prescription_idx
            prescription_dfs.append(pres_df)

        # Combine dfs for all prescriptions into a single df for the submission
        prescription_df = pd.concat(prescription_dfs)
        prescription_df = prescription_df.drop(columns='GeoID')

        return prescription_df

    def get_predictions(self, start_date_str, end_date_str, pres_df):
        start_date = pd.to_datetime(start_date_str)
        last_known_date = self.predictor.df['Date'].max()
        if last_known_date < pd.to_datetime(self.hist_df['Date'].min()) - np.timedelta64(1, 'D'):
            # append prior NPIs to the prescripted ones because the predictor will need them
            prior_ips_df = self.hist_df[(self.hist_df['Date'] > last_known_date) & (self.hist_df['Date'] < start_date)]
            prior_ips_df = prior_ips_df[pres_df.columns()]
            ips_df = pres_df.append(prior_ips_df)
        else:
            ips_df = pres_df

        # write ips_df to file
        ips_df.to_csv(TMP_PRESCRIPTION_FILE)

        # use full path of the local file passed as ip_file
        ip_file_full_path = os.path.abspath(TMP_PRESCRIPTION_FILE)

        # generate the predictions
        pred_df = self.predictor.predict(start_date_str, end_date_str, ips_df)

        return pred_df

if __name__ == '__main__':
    prescriptor = Neat(seed=42)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, 'prescriptions')
    ofile_path = os.path.abspath(os.path.join(output_dir, 'neat_evaluate_{}-action_dur-{}_checkpoint-{}_big_test.csv'.format(path, ACTION_DURATION, num_checkpoint)))
    print(ofile_path)
    print()
    prescriptor.evaluate(output_file_path=ofile_path)
