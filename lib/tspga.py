import numpy as np
import pandas as pd
from .city import *


class GAtsp:

    def __init__(self, population_size=10, best_parent=2, nextgen_parent=5, mutation_prob=0.5, tournament_size=5):
        self.population_size = population_size
        self.best_parent = best_parent
        self.nextgen_parent = nextgen_parent
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size

        self.print_log_step = 25

    def fit(self, city_df, distance_df):
        self.cities = city_df
        self.distances = distance_df

        # gene
        self.gene_length = len(self.cities)
        # city_id arr
        self.gene_arr = np.arange(self.gene_length)

    def run_ga(self, max_generation=100):
        self.population = self._init_population()
        self.fitness = self.compute_fitness()
        print("> generation: init")
        print("> fittest:", round(self.fitness[0], 4), round(1 / self.fitness[0], 2))
        for gen in range(1, max_generation + 1):
            self.fitness = self.compute_fitness()
            self.selection()
            self.breeding()
            self.population = self.nextgen
            if gen % self.print_log_step == 0 or gen == 1:
                print("generation:", gen)
                self.fitness = self.compute_fitness()
                print("fittest:", round(self.fitness[0], 4), round(1 / self.fitness[0], 2))
        self.fitness = self.compute_fitness()
        print("> generation: last")
        print("> fittest:", round(self.fitness[0], 4), round(1 / self.fitness[0], 2))

    def run_ga_save_log(self, max_generation=100, log_path="./ga_log/"):
        self.population = self._init_population()
        self.fitness = self.compute_fitness()
        print("> generation: init")
        print("> fittest:", round(self.fitness[0], 4), round(1 / self.fitness[0], 2))
        df = pd.DataFrame(self.population)
        df.to_csv(f"{log_path}/_ga_log_start.csv", index=False)
        for gen in range(1, max_generation + 1):
            self.fitness = self.compute_fitness()
            self.selection()
            self.breeding()
            self.population = self.nextgen
            df = pd.DataFrame(self.population)
            df.to_csv(f"{log_path}/ga_log_{gen}.csv", index=False)
            if gen % self.print_log_step == 0 or gen == 1:
                print("generation:", gen)
                self.fitness = self.compute_fitness()
                print("fittest:", round(self.fitness[0], 4), round(1 / self.fitness[0], 2))
        self.fitness = self.compute_fitness()
        print("> generation: last")
        print("> fittest:", round(self.fitness[0], 4), round(1 / self.fitness[0], 2))
        df = pd.DataFrame(self.population)
        df.to_csv(f"{log_path}/_ga_log_final.csv", index=False)

    def compute_fitness(self):
        population_distance = self._compute_distance()
        fitness = 1 / population_distance
        return np.array(fitness)

    def selection(self):
        self.nextgen = np.zeros_like(self.population)

        # fitness index (sort fitness by descending)
        fitness_sort_ind = np.argsort(self.fitness)[::-1]

        # select best n parent to next generation
        for i in range(self.best_parent):
            self.nextgen[i] = self.population[fitness_sort_ind[i]]

        # select parent to next generation by tournament selection
        for i in range(self.best_parent, self.nextgen_parent):
            tour_winner = self._tournament_selection()
            self.nextgen[i] = tour_winner

    def breeding(self):
        parent_pool = self.nextgen[:self.nextgen_parent]
        parent_id = np.arange(self.nextgen_parent)

        # crossover
        for i in range(self.nextgen_parent, self.population_size, 2):
            parent1, parent2 = parent_pool[np.random.choice(parent_id, 2, replace=False)]
            start, end = np.sort(np.random.choice(self.gene_arr, 2, replace=False))
            offspring1, offspring2 = self.crossover(parent1, parent2, start, end)
            self.nextgen[i] = offspring1
            if i + 1 < self.population_size:
                self.nextgen[i + 1] = offspring2
        for i in range(self.nextgen_parent, self.population_size):
            loc1, loc2 = np.sort(np.random.choice(self.gene_arr, 2, replace=False))
            self.nextgen[i] = self.mutation(self.nextgen[i], loc1, loc2)

    def mutation(self, inv, loc1, loc2):
        inv_copy = inv.copy()
        if np.random.rand() <= self.mutation_prob:
            inv_copy[loc1], inv_copy[loc2] = inv_copy[loc2], inv_copy[loc1]
        return inv_copy

    def crossover(self, parent1, parent2, start, end):
        parent1 = parent1.copy()
        parent2 = parent2.copy()

        p1_gene = parent1[start:end]
        p2_gene = parent2[start:end]

        offspring1 = parent1[~np.in1d(parent1, p2_gene)]
        offspring2 = parent2[~np.in1d(parent2, p1_gene)]

        for loc in range(start, end):
            offspring1 = np.insert(offspring1, loc, p2_gene[0])
            offspring2 = np.insert(offspring2, loc, p1_gene[0])

            p1_gene = np.delete(p1_gene, 0)
            p2_gene = np.delete(p2_gene, 0)

        return offspring1, offspring2

    def _init_population(self):
        population = []
        for i in range(self.population_size):
            population.append(np.random.choice(self.gene_arr, self.gene_length, replace=False))
        return np.array(population)

    def _compute_distance(self):
        population_distance = []
        for individual in self.population:
            population_distance.append(calculate_distance(self.distances, individual))
        return np.array(population_distance)

    def _tournament_selection(self):
        tour_candidates = np.random.choice(self.fitness, self.tournament_size, replace=True)
        tour_winner = np.max(tour_candidates)
        tour_winner_finess_ind = np.where(self.fitness == tour_winner)[0][0]

        fittest_inv = self.population[tour_winner_finess_ind]
        return fittest_inv
