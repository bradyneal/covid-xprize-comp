"""Implements the core evolution algorithm."""
from __future__ import print_function

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
from collections import defaultdict, namedtuple

import numpy as np


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome

    def dominates(self, one, other):
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

    def sortNondominatedNSGA2(self, k, first_front_only=False):
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
        for key in self.population:
            ind = self.population[key]
            map_fit_ind[ind.fitness_mult].append(ind)
        fits = list(map_fit_ind.keys())

        current_front = []
        next_front = []
        dominating_fits = defaultdict(int)
        dominated_fits = defaultdict(list)

        # Rank first Pareto front
        for i, fit_i in enumerate(fits):
            for fit_j in fits[i + 1:]:
                if self.dominates(fit_i, fit_j):
                    dominating_fits[fit_j] += 1
                    dominated_fits[fit_i].append(fit_j)
                elif self.dominates(fit_j, fit_i):
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
            N = min(len(self.population), k)
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

    def assignCrowdingDist(self, individuals):
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
            individuals[i].crowding_dist = dist/2
            pass
        pass



    def run_NSGA2(self, fitness_function, n=None, multi=False, algo='NSGA-2'):
        # algo possible values: NSGA-2, PO - pareto-optimality
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)
            # if multi-objective, set fitness
            if multi:
                if algo == 'NSGA-2':
                    # sort NSGA-2
                    fronts = self.sortNondominatedNSGA2(len(self.population))
                    pass
                # assign crowding distance
                for front in fronts:
                    self.assignCrowdingDist(front)
                # now assign fitness value
                num_fronts = len(fronts)
                for i in range(0, num_fronts):
                    front = fronts[i]
                    for el in front:
                        el.fitness = (num_fronts-i) + el.crowding_dist
                        pass
                    pass
                pass

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    self.reporters.end_generation(self.config, self.population, self.species)
                    break

            # Create the next generation from the current generation.
            new_population = self.reproduction.reproduce(self.config, self.species,
                                                         self.config.pop_size, self.generation)
            fitness_function(list(iteritems(new_population)), self.config)
            self.population = {**self.population, **new_population}
            # if multi-objective, set fitness
            if multi:
                if algo == 'NSGA-2':
                    # sort NSGA-2
                    fronts = self.sortNondominatedNSGA2(len(self.population))
                    pass
                # assign crowding distance
                for front in fronts:
                    self.assignCrowdingDist(front)
                # now assign fitness value
                num_fronts = len(fronts)
                for i in range(0, num_fronts):
                    front = fronts[i]
                    for el in front:
                        el.fitness = (num_fronts - i) + el.crowding_dist
                        pass
                    pass
                pass
            new_population = list(iteritems(self.population))
            new_population.sort(reverse=True, key=lambda x: x[1].fitness)
            self.population = dict(new_population[:self.config.pop_size])
            # fitness_function(list(iteritems(self.population)), self.config)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome
