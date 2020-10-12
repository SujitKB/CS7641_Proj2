import numpy as np
import mlrose_hiive as mlr_h
import mlrose
import matplotlib.pyplot as plt
import time
from random import randint

########################################## Knapsack

def knapsack_input_size():

    fitness_sa_arr = []
    fitness_rhc_arr = []
    fitness_ga_arr = []
    fitness_mimic_arr = []

    time_sa_arr = []
    time_rhc_arr = []
    time_ga_arr = []
    time_mimic_arr = []
    for n in range(20, 101, 20):

        fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, n),
                                 values=np.random.randint(30, 50, n),
                                 max_weight_pct=0.6)

        problem = mlr_h.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)

        print("input size ", n)

        init_state = np.random.randint(2, size=n)
        schedule = mlr_h.ExpDecay(init_temp=0.01)

        ############################################################################
        ########################### simulated_annealing ############################

        print("SA")
        st = time.time()

        schedule = mlr_h.ExpDecay(init_temp=0.1)
        best_state_sa, best_fitness_sa, fitness_curve_sa = mlr_h.simulated_annealing(problem,
                                                                                      schedule=schedule,
                                                                                      max_attempts=1000,
                                                                                      max_iters=10000,
                                                                                      init_state=init_state,
                                                                                      random_state=42,
                                                                                      curve=True)
        end = time.time()
        sa_time = end - st

        ############################################################################
        ############################ random_hill_climb #############################

        print("RHC")
        st = time.time()
        best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlr_h.random_hill_climb(problem,
                                                                                       restarts=8,
                                                                                       max_attempts=1000,
                                                                                       max_iters=10000,
                                                                                       init_state=init_state,
                                                                                       random_state=42,
                                                                                       curve=True)
        end = time.time()
        rhc_time = end - st

        ############################################################################
        ################################# genetic_alg ##############################

        print("GA")
        st = time.time()
        best_state_ga, best_fitness_ga, fitness_curve_ga = mlr_h.genetic_alg(problem,
                                                                              pop_size=100,
                                                                              mutation_prob=0.01,
                                                                              max_attempts=100,
                                                                              max_iters=5000,
                                                                              random_state=42,
                                                                              curve=True)
        end = time.time()
        ga_time = end - st

        ############################################################################
        #################################### mimic #################################

        print("MIMIC")
        st = time.time()
        best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlr_h.mimic(problem,
                                                                                 pop_size=500,
                                                                                 keep_pct=0.3,
                                                                                 max_attempts=300,
                                                                                 max_iters=10000,
                                                                                 random_state=42,
                                                                                 curve=True)
        end = time.time()
        mimic_time = end - st

        fitness_sa_arr.append(best_fitness_sa)
        fitness_rhc_arr.append(best_fitness_rhc)
        fitness_ga_arr.append(best_fitness_ga)
        fitness_mimic_arr.append(best_fitness_mimic)

        time_sa_arr.append(sa_time)
        time_rhc_arr.append(rhc_time)
        time_ga_arr.append(ga_time)
        time_mimic_arr.append(mimic_time)

    fitness_sa_arr = np.array(fitness_sa_arr)
    fitness_rhc_arr = np.array(fitness_rhc_arr)
    fitness_ga_arr = np.array(fitness_ga_arr)
    fitness_mimic_arr = np.array(fitness_mimic_arr)

    time_sa_arr = np.array(time_sa_arr)
    time_rhc_arr = np.array(time_rhc_arr)
    time_ga_arr = np.array(time_ga_arr)
    time_mimic_arr = np.array(time_mimic_arr)

    plt.figure()
    plt.plot(np.arange(20, 101, 20), fitness_sa_arr, label='SA')
    plt.plot(np.arange(20, 101, 20), fitness_rhc_arr, label='RHC')
    plt.plot(np.arange(20, 101, 20), fitness_ga_arr, label='GA')
    plt.plot(np.arange(20, 101, 20), fitness_mimic_arr, label='MIMIC')
    plt.xlabel('Bit String Size')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.grid()
    plt.title('Fitness Value v/s Bit String Size (Continuous Peaks)')
    plt.savefig('knapsackInputSizeFitness.png')
    plt.clf()

    plt.figure()
    plt.plot(np.arange(20, 101, 20), time_sa_arr, label='SA')
    plt.plot(np.arange(20, 101, 20), time_rhc_arr, label='RHC')
    plt.plot(np.arange(20, 101, 20), time_ga_arr, label='GA')
    plt.plot(np.arange(20, 101, 20), time_mimic_arr, label='MIMIC')
    plt.legend()
    plt.grid()
    plt.xlabel('Bit String Size')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time v/s Bit String Size (Continuous Peaks)')
    plt.savefig('knapsackInputSizeComputation.png')
    plt.clf()

def knapsack_iterations():

    n=100
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, n),
                             values=np.random.randint(30, 50, n),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)

    init_state = np.random.randint(2, size=n)
    schedule = mlr_h.ExpDecay(init_temp=0.1)
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlr_h.simulated_annealing(problem, schedule=schedule,
                                                                                  max_attempts=1000, max_iters=10000,
                                                                                  init_state=init_state,
                                                                                  random_state=42,
                                                                                  curve=True)


    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlr_h.random_hill_climb(problem, restarts=8,
                                                                                   max_attempts=1000,
                                                                                   max_iters=10000,
                                                                                   init_state=init_state,
                                                                                   random_state=42,
                                                                                   curve=True)

    best_state_ga, best_fitness_ga, fitness_curve_ga = mlr_h.genetic_alg(problem, pop_size=100,
                                                                                    mutation_prob=0.01,
                                                                                    max_attempts=50, max_iters=5000,
                                                                                    random_state=42,
                                                                                    curve=True)

    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlr_h.mimic(problem, pop_size=500,
                                                                             keep_pct=0.3,
                                                                             max_attempts=300,
                                                                             max_iters=10000,
                                                                             random_state=42,
                                                                             curve=True)

    plt.figure()
    plt.plot(fitness_curve_sa, label='SA')
    plt.plot(fitness_curve_rhc, label='RHC')
    plt.plot(fitness_curve_ga, label='GA')
    plt.plot(fitness_curve_mimic, label='MIMIC')
    plt.legend()
    plt.grid()
    plt.ylabel('Fitness Value')
    plt.xlabel('Iterations#')
    plt.title('Fitness Value v/s Iterations# (Continuous Peaks)')
    plt.savefig('knapsackIterations.png')
    plt.clf()
    return

def knapsack_sa():
    knapsack_len = 100

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    schedule = mlr_h.ExpDecay(init_temp=0.01)
    best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlr_h.simulated_annealing(problem, schedule=schedule,
                                                                                        max_attempts=1000,
                                                                                        max_iters=10000,
                                                                                        init_state=init_state,
                                                                                        random_state=42,
                                                                                        curve=True)

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    schedule = mlr_h.ExpDecay(init_temp=0.1)
    best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlr_h.simulated_annealing(problem, schedule=schedule,
                                                                                        max_attempts=1000,
                                                                                        max_iters=10000,
                                                                                        init_state=init_state,
                                                                                        random_state=42,
                                                                                        curve=True)

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    schedule = mlr_h.ExpDecay(init_temp=1.0)
    best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlr_h.simulated_annealing(problem, schedule=schedule,
                                                                                        max_attempts=1000,
                                                                                        max_iters=10000,
                                                                                        init_state=init_state,
                                                                                        random_state=42,
                                                                                        curve=True)

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    schedule = mlr_h.ExpDecay(init_temp=10.0)
    best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlr_h.simulated_annealing(problem, schedule=schedule,
                                                                                        max_attempts=1000,
                                                                                        max_iters=10000,
                                                                                        init_state=init_state,
                                                                                        random_state=42,
                                                                                        curve=True)


    plt.figure()
    plt.plot(fitness_curve_sa_1, label='ExpDecay(init_temp=0.01)')
    plt.plot(fitness_curve_sa_2, label='ExpDecay(init_temp=0.1)')
    plt.plot(fitness_curve_sa_3, label='ExpDecay(init_temp=1.0)')
    plt.plot(fitness_curve_sa_4, label='ExpDecay(init_temp=10)')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Fitness Value')
    plt.title('Knapsack - SA - HyperParam Comparison')
    plt.savefig('knapsack_SA.png')
    plt.clf()

def knapsack_rhc():

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    best_state_rhc_1, best_fitness_rhc_1, fitness_curve_rhc_1 = mlr_h.random_hill_climb(problem, restarts=0,
                                                                                        max_attempts=1000,
                                                                                        max_iters=10000,
                                                                                        init_state=init_state,
                                                                                        random_state=42,
                                                                                        curve=True)
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)

    best_state_rhc_2, best_fitness_rhc_2, fitness_curve_rhc_2 = mlr_h.random_hill_climb(problem, restarts=2,
                                                                                         max_attempts=1000,
                                                                                         max_iters=10000,
                                                                                         init_state=init_state,
                                                                                         random_state=42,
                                                                                         curve=True)

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    best_state_rhc_3, best_fitness_rhc_3, fitness_curve_rhc_3 = mlr_h.random_hill_climb(problem, restarts=4,
                                                                                         max_attempts=1000,
                                                                                         max_iters=10000,
                                                                                         init_state=init_state,
                                                                                         random_state=42,
                                                                                         curve=True)

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    best_state_rhc_4, best_fitness_rhc_4, fitness_curve_rhc_4 = mlr_h.random_hill_climb(problem, restarts=6,
                                                                                         max_attempts=1000,
                                                                                         max_iters=10000,
                                                                                         init_state=init_state,
                                                                                         random_state=42,
                                                                                         curve=True)

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    best_state_rhc_5, best_fitness_rhc_5, fitness_curve_rhc_5 = mlr_h.random_hill_climb(problem, restarts=8,
                                                                                         max_attempts=1000,
                                                                                         max_iters=10000,
                                                                                         init_state=init_state,
                                                                                         random_state=42,
                                                                                         curve=True)

    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=100)
    best_state_rhc_6, best_fitness_rhc_6, fitness_curve_rhc_6 = mlr_h.random_hill_climb(problem, restarts=10,
                                                                                         max_attempts=1000,
                                                                                         max_iters=10000,
                                                                                         init_state=init_state,
                                                                                         random_state=42,
                                                                                         curve=True)

    plt.figure()
    plt.plot(fitness_curve_rhc_1, label='restarts=0')
    plt.plot(fitness_curve_rhc_2, label='restarts=2')
    plt.plot(fitness_curve_rhc_3, label='restarts=4')
    plt.plot(fitness_curve_rhc_4, label='restarts=6')
    plt.plot(fitness_curve_rhc_5, label='restarts=8')
    plt.plot(fitness_curve_rhc_6, label='restarts=10')
    plt.title('Knapsack - RHC - HyperParam Comparison')
    plt.legend()
    plt.grid()
    plt.xlabel('Iterations#')
    plt.ylabel('Fitness Value')
    plt.savefig('knapsackRhcAnalysis.png')
    plt.clf()

def knapsack_mimic():
    print("0")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)

    best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlr_h.mimic(problem, keep_pct=0.1, pop_size=200,
                                                                          max_attempts=50, max_iters=10000,
                                                                          random_state=42,
                                                                          curve=True)
    print('4', best_fitness_sa_4)

    best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlr_h.mimic(problem, keep_pct=0.2, pop_size=200,
                                                                          max_attempts=50, max_iters=10000,
                                                                          random_state=42,
                                                                          curve=True)
    print('5', best_fitness_sa_5)

    best_state_sa_6, best_fitness_sa_6, fitness_curve_sa_6 = mlr_h.mimic(problem, keep_pct=0.3, pop_size=200,
                                                                          max_attempts=50, max_iters=10000,
                                                                          random_state=42,
                                                                          curve=True)
    print('6', best_fitness_sa_6)

    best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlr_h.mimic(problem, keep_pct=0.1, pop_size=500,
                                                                          max_attempts=50, max_iters=10000,
                                                                          random_state=42,
                                                                          curve=True)
    print('7', best_fitness_sa_7)

    best_state_sa_8, best_fitness_sa_8, fitness_curve_sa_8 = mlr_h.mimic(problem, keep_pct=0.2, pop_size=500,
                                                                          max_attempts=100, max_iters=10000,
                                                                          random_state=42,
                                                                          curve=True)
    print('8', best_fitness_sa_8)

    best_state_sa_9, best_fitness_sa_9, fitness_curve_sa_9 = mlr_h.mimic(problem, keep_pct=0.3, pop_size=500,
                                                                          max_attempts=100, max_iters=10000,
                                                                          random_state=42,
                                                                          curve=True)
    print('9', best_fitness_sa_9)
    plt.figure()
    plt.plot(fitness_curve_sa_4, label='0.1 & 200')
    plt.plot(fitness_curve_sa_5, label='0.2 & 200')
    plt.plot(fitness_curve_sa_6, label='0.3 & 200')
    plt.plot(fitness_curve_sa_7, label='0.1 & 500')
    plt.plot(fitness_curve_sa_8, label='0.2 & 500')
    plt.plot(fitness_curve_sa_9, label='0.3 & 500')
    plt.title('Knapsack - MIMIC - HyperParam Comparison')
    plt.legend()
    plt.grid()
    plt.xlabel('Iterations#')
    plt.ylabel('Fitness Value')
    plt.savefig('knapsackMimicAnalysis.png')
    plt.clf()

def knapsack_ga():
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)

    print("1")
    best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlr_h.genetic_alg(problem, mutation_prob=0.01,
                                                                                pop_size=100, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)
    print("2")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlr_h.genetic_alg(problem, mutation_prob=0.1,
                                                                                pop_size=100, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)
    print("3")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlr_h.genetic_alg(problem, mutation_prob=0.2,
                                                                                pop_size=100, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)
    print("4")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlr_h.genetic_alg(problem, mutation_prob=0.01,
                                                                                pop_size=200, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)
    print("5")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlr_h.genetic_alg(problem, mutation_prob=0.1,
                                                                                pop_size=200, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)
    print("6")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    best_state_sa_6, best_fitness_sa_6, fitness_curve_sa_6 = mlr_h.genetic_alg(problem, mutation_prob=0.2,
                                                                                pop_size=200, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)
    print("7")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlr_h.genetic_alg(problem, mutation_prob=0.01,
                                                                                pop_size=500, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)
    print("8")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)

    best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlr_h.genetic_alg(problem, mutation_prob=0.1,
                                                                                pop_size=500, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)
    print("9")
    fitness = mlr_h.Knapsack(weights=np.random.randint(20, 50, 100),
                             values=np.random.randint(30, 50, 100),
                             max_weight_pct=0.6)

    problem = mlr_h.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)

    best_state_sa_8, best_fitness_sa_8, fitness_curve_sa_8 = mlr_h.genetic_alg(problem, mutation_prob=0.2,
                                                                                pop_size=500, max_attempts=100,
                                                                                max_iters=5000,
                                                                                random_state=42,
                                                                                curve=True)

    plt.figure()
    plt.plot(fitness_curve_sa_1, label='0.01 & 100')
    plt.plot(fitness_curve_sa_2, label='0.1 & 100')
    plt.plot(fitness_curve_sa_3, label='0.2 & 100')
    plt.plot(fitness_curve_sa_4, label='0.01 & 200')
    plt.plot(fitness_curve_sa_5, label='0.1 & 200')
    plt.plot(fitness_curve_sa_6, label='0.2 & 200')
    plt.plot(fitness_curve_sa_7, label='0.01 & 500')
    plt.plot(fitness_curve_sa_8, label='0.1 & 500')
    plt.plot(fitness_curve_sa_8, label='0.2 & 500')
    plt.title('Knapsack - GA - HyperParam Comparison')
    plt.legend()
    plt.grid()
    plt.xlabel('Iterations#')
    plt.ylabel('Fitness Value')
    plt.savefig('knapsackGaAnalysis.png')
    plt.clf()


############################################ Knapsack main block #######################################
knapsack_input_size()
knapsack_iterations()
knapsack_sa()
knapsack_ga()
knapsack_rhc()
knapsack_mimic()
