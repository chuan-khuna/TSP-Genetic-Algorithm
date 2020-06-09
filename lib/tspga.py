
import numpy as np
import pandas as pd

from .city import *
from .visualize import *

class TSPGA:
    def __init__(self, city_df, distance_df, pop_size=5, max_generation=10, nextgen_num_parent=2, tournament_size=5, mutation_prob=0.5, mutation_lim=2):

        # data frame
        self.city_df = city_df
        self.distance_df = distance_df

        # ga setting
        self.pop_size = pop_size
        self.gene_size = self.city_df.shape[0]
        self.max_generation = max_generation

        # next gen setting
        # number of parent to nextgen
        self.nextgen_num_parent = nextgen_num_parent
        self.tournament_size =tournament_size
        self.mutation_prob = mutation_prob
        self.mutation_lim = mutation_lim

        self.pop = self.init_pop()

    def run_ga(self, log=False, output_path='./ga_log/'):

        if log:
            df = pd.DataFrame(self.pop)
            df.to_csv(f'{output_path}/_gen_init.csv', index=False)
            
            for generation in range(1, self.max_generation+1):
                self.assign_fitness()
                print(f"{generation=}")
                df = pd.DataFrame(self.pop)
                df.to_csv(f'{output_path}/gen_{generation}.csv', index=False)

                self.selection()
                self.create_offspring()
            
            df = pd.DataFrame(self.pop)
            df.to_csv(f'{output_path}/_gen_last.csv', index=False)

        else:
            for generation in range(1, self.max_generation+1):
                print(f"{generation=}")
                self.assign_fitness()
                self.selection()
                self.create_offspring()

    def init_pop(self):
        population = []
        route = np.arange(0, self.gene_size)
        for i in range(self.pop_size):
            population.append(np.random.choice(route, len(route), replace=False))
        return np.array(population)


    def assign_fitness(self):
        self.pop_distance = self._cal_population_distance()
        self.pop_fitness = 1/self.pop_distance


    def selection(self):
        self.next_gen = np.zeros_like(self.pop)
        
        # send fittest parent to next gen
        self.next_gen[0] = self.pop[np.argmax(self.pop_fitness)]

        for i in range(1, self.nextgen_num_parent):
            self.next_gen[i] = self._tournament_selection(self.tournament_size)


    def create_offspring(self):
        parent_ind = np.arange(0, self.nextgen_num_parent)
        gene_ind = np.arange(0, self.gene_size)

        # cross over
        for i in range(self.nextgen_num_parent, self.pop_size):
            selected_parent =  np.random.choice(parent_ind, 2, replace=False)
            cross_over_loc = np.sort(np.random.choice(gene_ind, 2, replace=False))
            self.next_gen[i] = self._cross_over(self.next_gen[selected_parent[0]], self.next_gen[selected_parent[1]], cross_over_loc[0], cross_over_loc[1])
        
        # mutation
        for i in range(self.nextgen_num_parent, self.pop_size):
            # gaurantee mutation
            mutation_loc = np.sort(np.random.choice(gene_ind, 2, replace=False))
            self.next_gen[i] = self._mutation(self.next_gen[i], mutation_loc[0], mutation_loc[1])

            # mutation again
            for m in range(self.mutation_lim):
                if np.random.rand() < self.mutation_prob:
                    mutation_loc = np.sort(np.random.choice(gene_ind, 2, replace=False))
                    self.next_gen[i] = self._mutation(self.next_gen[i], mutation_loc[0], mutation_loc[1])
                else:
                    self.next_gen[i] = self.next_gen[i]
                    break

        self.pop = self.next_gen

    def _cal_population_distance(self):
    
        population_distance = []

        for invidual in self.pop:
            population_distance.append(calculate_distance(self.distance_df, invidual))

        return np.array(population_distance)

    def _tournament_selection(self, tour_size):
        tournament_candidate_fitness = []
        for i in range(tour_size):
            tournament_candidate_fitness.append(np.random.choice(self.pop_fitness))
        tournament_fittest = np.max(tournament_candidate_fitness)

        tournament_fittest_inv = self.pop[np.where(self.pop_fitness == tournament_fittest)[0][0]]
        return tournament_fittest_inv
    
    def _cross_over(self, parent1, parent2, start, end):
        p1_gene = parent1[start:end]
        # select gene that not in parent1
        p2_gene = parent2[~np.in1d(parent2, p1_gene)]

        offspring = p2_gene
        for i in range(start, end):
            offspring = np.insert(offspring, i, p1_gene[0])
            p1_gene = np.delete(p1_gene, 0)

        return offspring

    def _mutation(self, offspring, loc1, loc2):
        offspring[loc1], offspring[loc2] = offspring[loc2], offspring[loc1]
        return offspring