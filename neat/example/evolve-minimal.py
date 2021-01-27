"""
2-input XOR example -- this is most likely the simplest possible example.
Description https://neat-python.readthedocs.io/en/latest/xor_example.html
"""

from __future__ import print_function
import neat
import numpy as np

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        if genome.fitness is not None:
            # already computed, we don't have to do it again
            continue
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2
            pass
    pass


def eval_genomes_2d(genomes, config):
    for genome_id, genome in genomes:
        if genome.fitness is not None:
            # already computed, we don't have to do it again
            continue
        fitness_val = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            fitness_val -= (output[0] - xo[0]) ** 2
        if fitness_val < 3.5:
            val_2 = abs(np.random.normal(0, 0.1))
            pass
        else:
            val_2 = 1./fitness_val
            pass
        # fitness will be computed later, during sorting
        genome.fitness = None
        # we set many-objective fitness instead, these are the values of 2 objectives, but we can have more
        # one objective per NPI is also possible
        genome.fitness_mult = (fitness_val, val_2)


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Add checkpointer to save population every generation and every 10 minutes.
p.add_reporter(neat.Checkpointer(generation_interval=1,
                                 # time_interval_seconds=600,
                                 filename_prefix='neat-checkpoint-'))

# Run until a solution is found.
#winner = p.run(eval_genomes)


# algo=PO     - recommended when number of objectives >=4
# algo=NSGA-2 - recommended when number of objectives is 2,3
# but we can also try both algorithms, these are general recommendations
# n - number of generations (better set number of generations as we don't have stopping criterion
# if multi=False - standard 1-objective optimization from Neat
#winner = p.run_NSGA2(eval_genomes_2d, n=150, multi=True, algo='PO')
winner = p.run_NSGA2(eval_genomes_2d, n=150, multi=True, algo='NSGA-2')

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
