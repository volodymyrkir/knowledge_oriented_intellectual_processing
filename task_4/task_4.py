import random
import numpy as np
from deap import base, creator, tools, algorithms

IND_SIZE = 50
POP_SIZE = 100
CXPB = 0.6
MUTPB = 0.02
NGEN = 100
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


toolbox.register("attr_bool", random.randint, 0, 1)


toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_one_max(individual):
    return (sum(individual),)


toolbox.register("evaluate", eval_one_max)


toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=MUTPB)


def run_ga(seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=CXPB, mutpb=0.0, ngen=NGEN,
        stats=stats, halloffame=hof, verbose=verbose
    )
    return pop, logbook, hof


if __name__ == "__main__":
    pop, logbook, hof = run_ga(seed=RANDOM_SEED, verbose=True)
    best = hof[0]
    print('\nНайкращий індивід:', best)
    print('Фітнес найкращого:', best.fitness.values[0])