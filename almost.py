# Importing 

from collections import defaultdict
import time
import numpy as np 
import genetic
import sys
import copy
import random



# Functions
class Genetic_Algorithm():
    def __init__(self, schedule, num_vertices, distance_matrix, num_par, num_pass, capacity, q):
        # num_vertices = num_cities
        self.schedule = schedule
        self.num_vertices = num_vertices
        self.vertices = [int(i) for i in range(num_vertices)]
        self.distance_matrix = distance_matrix
        self.num_pass = num_pass
        self.num_par = num_par
        self.capacity = capacity
        self.q = q
        # q is a list present the weight of all parcels

    def return_true_config(self, config):
        true_config = []
        for i in config:
            true_config.append(i)
            if i >= 1 and i <= self.num_pass + self.num_par:
                true_config.append(i + self.num_par + self.num_pass)
        return true_config

    def check_capacity(self, config):
        cap = 0
        for i in range(1, len(config)):
            if config[i] > self.num_pass and config[i] <= self.num_pass + self.num_par:
                cap += self.q[config[i] - self.num_pass - 1]
            elif config[i] > 2 * self.num_pass + self.num_par:
                cap -= self.q[config[i] - 2 * self.num_pass - self.num_par - 1]
            if cap > self.capacity:
                return False
        return True

    def cal_distance(self, config):
        # print("cal_distance_step:\n", config)
        if self.check_capacity(config) == False:
            return (-1)

        cost = 0
        explore = []
        for city in range(1, len(config)):
            explore.append(config[city])
            cost += self.distance_matrix[config[city - 1]][config[city]]
            if config[city] > 2 * self.num_pass + self.num_par:
                if (config[city] - self.num_pass - self.num_par) not in explore:
                    return (-1)
        return cost

    # * True
    def compute_capacity(self, config):
        cap = 0
        for x in config:
            if x > self.num_pass and x <= self.num_pass + self.num_par:
                cap += self.q[x - self.num_pass - 1]
            elif x > 2 * self.num_pass + self.num_par:
                cap -= self.q[x - 2 * self.num_pass - self.num_par - 1]
        return cap

    def children(self, list_node, state):
        curCap = self.compute_capacity(state)
        res = []
        for n in list_node:
            if n not in state:
                if curCap > self.capacity:
                    if (n - self.num_pass - self.num_par) in state:
                        res.append(n)
                else:
                    if n > self.num_pass + self.num_par:
                        if (n - self.num_pass - self.num_par) in state:
                            res.append(n)
                        else:
                            continue
                    else:
                        res.append(n)
        return res

    def generateValidState(self):
        state = [0]
        list_node = [self.schedule[i][0] for i in range(len(self.schedule))]

        curCap = 0
        while True:
            list_next_cities = self.children(list_node, state)
            if len(list_next_cities) == 0: break
            next_city = np.random.choice(list_next_cities, 1)[0]

            if next_city > self.num_pass and next_city <= self.num_pass + self.num_par:
                curCap += self.q[next_city - self.num_pass - 1]
            elif next_city > 2 * self.num_pass + self.num_par:
                curCap -= self.q[next_city - 2 * self.num_pass - self.num_par - 1]
            state.append(next_city)
        state.append(0)
        return state

    def swap_pos(self, config, pos1, pos2):
        config[pos1], config[pos2] = config[pos2], config[pos1]
        return config

    def satisfied_config(self, config):
        check = [[0, 0] for i in range(self.num_par + self.num_pass)]
        new_config = copy.deepcopy(config)

        for node in range(1, len(config) - 1):
            id = config[node]
            if id > self.num_pass + self.num_par:
                check[id - self.num_pass - self.num_par - 1][1] = node
            else:
                check[id - 1][0] = node
        for loc in check:
            if loc[0] > loc[1]:
                new_config = self.swap_pos(new_config, loc[0], loc[1])
        return new_config

    def crossover(self, par1, par2):
        genA = np.random.choice(np.arange(1, len(par1) - 1), size=1)[0]
        genB = np.random.choice(np.arange(1, len(par1) - 1), size=1)[0]
        if genA < genB:
            pos1, pos2 = genA, genB
        else:
            pos1, pos2 = genB, genA
        temp1 = []
        for i in range(pos1, pos2):
            temp1.append(par1[i])
        cnt = 0
        temp2 = []
        for i in par2:
            if i not in temp1 and cnt < pos1:
                temp2.append(i)
                cnt += 1
        temp3 = [item for item in par2 if item not in temp1 and item not in temp2]
        child1 = temp2 + temp1 + temp3
        child2 = self.satisfied_config(child1)
        return child1, child2

    def mutation(self, config):
        check = [[0, 0] for i in range(self.num_par + self.num_pass)]
        new_config = [0] * len(config)

        for node in range(1, len(config) - 1):
            id = config[node]
            if id > self.num_pass + self.num_par:
                check[id - self.num_pass - self.num_par - 1][1] = node
            else:
                check[id - 1][0] = node
        chromosome_1 = np.random.randint(0, len(check) - 1)
        chromosome_2 = np.random.randint(0, len(check) - 1)
        check = self.swap_pos(check, chromosome_1, chromosome_2)
        for id_chr in range(len(check)):
            new_config[check[id_chr][0]] = id_chr + 1
            new_config[check[id_chr][1]] = id_chr + 1 + self.num_par + self.num_pass
        return new_config

    def solving_gene(self, maxIter=15, num_genes=40, crossover_rate=0.4, mutation_rate=0.1, num_elites=5):
        population = []
        while len(population) != num_genes:
            initial_gene = self.generateValidState()
            population.append([initial_gene, self.cal_distance(self.return_true_config(initial_gene))])
        # print("\n\nGene step")
        population.sort(key=lambda x: x[1])
        elites_pop = copy.deepcopy(population[:num_elites])
        cur_opt_cost = 0
        cur_opt_config = []
        cnt = 0
        i = 0
        while i < maxIter:
            i += 1
            j = 0
            new_population = []
            while j < num_genes // 4:
                j += 1
                for elite in elites_pop:
                    # * We should try to crossover, then mutate
                    cross = random.random()
                    if cross > crossover_rate:
                        # crossover randomly
                        cnt += 1
                        par1 = np.random.choice(np.arange(len(elites_pop)), size=1)[0]
                        par2 = np.random.choice(np.arange(len(elites_pop)), size=1)[0]
                        child1, child2 = self.crossover(elites_pop[par1][0], elites_pop[par2][0])
                        child1_cost = self.compute_capacity(self.return_true_config(child1))
                        child2_cost = self.compute_capacity(self.return_true_config(child2))
                        new_population.append([child1, child1_cost])
                        new_population.append([child2, child2_cost])

                    mutate = random.random()
                    if mutate < mutation_rate:
                        new_child = self.mutation(elite[0])
                        child_cost = self.compute_capacity(self.return_true_config(new_child))
                        new_population.append([new_child, child_cost])

            population.sort(key=lambda x: x[1])
            elites_pop.extend(population[:num_elites])
            elites_pop.sort(key=lambda x: x[1])
            elites_pop = copy.deepcopy(elites_pop[:num_elites])

            cur_opt_cost = elites_pop[0][1]
            cur_opt_config = elites_pop[0][0]
        return cur_opt_cost, cur_opt_config

def random_configuration(num_cars, num_par, num_pass, iter):
    def random_conf():
        config = defaultdict(lambda: [0])
        pair_config = defaultdict(lambda: [[0, -1]])
        for items in range(1, num_pass + num_par):
            car = np.random.choice(num_cars, 1)[0]
            if items > num_pass and items <= num_pass + num_par:
                config[car + 1].append([items])
                config[car + 1].append([items + num_par + num_pass])
            elif items >= 1 and items <= num_pass:
                config[car + 1].append([items])
        for i in range(1, num_cars + 1):
            temp = list()
            for j in config[i]:
                if type(j) == list:
                    temp.append(j[0])
            sorted(temp)
            for j in temp:
                if j >= 1 and j <= num_pass:
                    pair_config[i].append([j, j + num_par + num_pass])
                else:
                    pair_config[i].append([j, -1])
        return pair_config
    
    #* when meet a passenger, just take him, we can temporarily ignore he/she's destination
    def uniform_random_conf():
        pair_config = defaultdict(lambda: [[0, -1]])
        arr = np.arange(1, num_pass + num_par + 1)
        np.random.shuffle(arr)
        split_arr = np.array_split(arr, num_cars)
        for car in range(num_cars):
            single_pair_conf = list()
            for items in split_arr[car]:
                if items > num_pass and items <= num_pass + num_par:
                    single_pair_conf.append([items, -1])
                    single_pair_conf.append([items + num_par + num_pass, -1])
                elif items >= 1 and items <= num_pass:
                    single_pair_conf.append([items, items + num_par + num_pass])
            pair_config[car + 1].extend(sorted(single_pair_conf))
        return pair_config

    if iter&1:
        return uniform_random_conf()
    else:
        return random_conf()

def return_true_config(given_config, num_pass, num_par):
    true_config = []
    for i in given_config:
        true_config.append(i)
        if i >= 1 and i <= num_pass:
            true_config.append(i + num_par + num_pass)
    return true_config
    
    
def cal_distance(config, distance_matrix, num_pass, num_par):
        cost = 0
        explore = []
        for city in range(1, len(config)):
            explore.append(config[city])
            cost += distance_matrix[config[city - 1]][config[city]]
            if config[city] > 2*num_pass + num_par:
                if (config[city] - num_pass - num_par) not in explore:
                    return (-1)
        return cost
            
# Main and input

"""
N, M, K = map(int, input().split())
q = [int(x) for x in input().split()] # size = M
Q = [int(x) for x in input().split()] # size = K
d = [list(map(int, input().split())) for _ in range(2*N + 2*M + 1)] # size = (2*N + 2*M + 1)*(2*N + 2*M + 1)
"""
# Read input from file


s = 1
with open(f"test case\\test{s}.txt", "r") as file:
    N, M, K = map(int, file.readline().split())
    q = list(map(int, file.readline().split()))  # size = M
    Q = list(map(int, file.readline().split()))  # size = K
    d = [list(map(int, file.readline().split())) for _ in range(2 * N + 2 * M + 1)]  # size = (2 * N + 2 * M + 1) * (2 * N + 2 * M + 1)

if __name__ == "__main__":
    num_cities = 2*N + 2*M + 1 
    num_cars = K
    cars_capacities = Q
    
    matrix_distance = d
    num_pass_par = N + M
    num_pass = N
    num_par = M

    # sys.stdout = open(f".\\result\\ga\\test{s}.txt", "w")
    print("Genetic Algorithm")
    ga_start_time = time.time()

    # matrix distance is correct

    final_res = 1e9
    final_res_config = []

    for iter in range(16):
        # print(f"iter {iter}:")
        res_conf = []
        dict_schedule = random_configuration(num_cars, num_par, num_pass, iter)
        max_res_each_config = 0
        for id_bus, schedule in dict_schedule.items():
            ga = genetic.Genetic_Algorithm(schedule, num_cities, matrix_distance, num_pass, num_pass, cars_capacities[id_bus - 1], q)
            # print("Initial schedule is: ", schedule)
            temp_res = ga.solving_gene()

            res_conf.append(temp_res)
            temp = cal_distance(return_true_config(temp_res[1], num_pass, num_par), matrix_distance, num_pass, num_par)
            if max_res_each_config < temp:
                max_res_each_config = temp
            # print(temp_res[1], temp_res[0], sep = " - ", end='\n')
        if final_res > max_res_each_config:
            final_res = max_res_each_config
            final_res_config = res_conf
            # print(final_res)

    print(num_cars)
    for conf in final_res_config:
        true_config = return_true_config(conf[1], num_pass, num_par)
        print(len(true_config))
        for i in true_config:
            print(i, end = " ")
        print(f"- {cal_distance(true_config, matrix_distance, num_pass, num_par)}")
    print(final_res)


    total_running_time = time.time() - ga_start_time
    print(f"Running time = {total_running_time}")

    # DONE WITH GENETIC ALGORITHM