import numpy as np
from sklearn.metrics import mean_squared_error
import random
from functools import partial
from tqdm import tqdm
import time



class GeneticAlgorithm:
    '''This class implements a genetic algorithm.
    '''

    def __init__(
            self, 
            df,
            columns : list
        ):
        '''Initializes the genetic algorithm class.

        Parameters
        ----------
        df : pandas DataFrame
            The dataframe to be used in the algorithm.
        columns : list
            The columns of the dataframe that will be used in the algorithm.
        '''
        self.df = df.copy()
        self.columns = columns
    


    def generate_individual(
            self, 
            lenght : int
        ):
        '''Generates an individual (i.e. a possible solution) for the optimization problem.

        Parameters
        ----------
        lenght : int
            The length of the individual.

        Returns
        -------
        numpy.ndarray
            A numpy array of shape (lenght,) containing random uniform values between 0 and 1.
        '''
        return np.random.uniform(0, 1, lenght)



    def generate_population(
            self,
            pop_size: int,
            individual_length : int
        ):
        '''Generates a population (i.e. a set of possible solutions) for the optimization problem.

        Parameters
        ----------
        pop_size : int
            The number of individuals in the population.
        individual_length : int
            The length of each individual in the population.

        Returns
        -------
        list[numpy.ndarray]
            A list of numpy arrays of shape (individual_length,) 
            each of them of shape (lenght,) containing random uniform values between 0 and 1.
        '''
        return [self.generate_individual(individual_length) for _ in range(pop_size)]
    

    
    def fitness_pandas_apply(
            self,
            individual
        ):
        '''This is the first version of the fitness function. 
           Its runtime is around 0.22-0.23 seconds on a DateFrame of 3620 rows, 
           population size 40 and mutation_prob 0.4.
        '''
        start_time = time.time()

        mse = self.df[self.columns].apply(lambda x: mean_squared_error(individual, x, squared=True), axis=1)
        min_index = mse.idxmin()

        execution_time = time.time() - start_time
        print("Execution time:", execution_time)

        return mse[min_index] * self.df.iloc[min_index]['swarming_score']



    def fitness_apply_along_axis(
            self,
            individual
        ):
        '''This is the second version of the fitness function. 
           Its runtime is around 0.17-0.19 seconds on a DateFrame of 3620 rows, 
           population size 40 and mutation_prob 0.4.
        '''
        start_time = time.time()

        mse = np.apply_along_axis(lambda x: mean_squared_error(individual, x, squared=True), axis=1, arr=self.df[self.columns].values)
        min_index = np.argmin(mse)

        execution_time = time.time() - start_time
        print("Execution time:", execution_time)

        return mse[min_index] * self.df.iloc[min_index+1]['swarming_score']



    def fitness(
            self,
            individual
        ):
        '''Evaluates the fitness (i.e. the quality) of a given individual in the optimization problem.

        Parameters
        ----------
        individual : numpy.ndarray
             The individual to be evaluated. It must have the same length as the columns of the dataframe.

        Returns
        -------
        float
            The fitness value of the individual, which is a measure of how well the individual performs in the optimization problem.
        '''
        #start_time = time.time()
        df_diff = self.df[self.columns].sub(individual, axis=1)
        df_diff_squared = df_diff.pow(2)
        mse = df_diff_squared.mean(axis=1)
        min_index = mse.idxmin()
        
        #execution_time = time.time() - start_time
        #print("Execution time:", execution_time)

        return mse[min_index] * self.df.iloc[min_index]['swarming_score']



    def roulette_selection(
            self, 
            population, 
            fitness
        ):
        '''Selects two parents from a population using the roulette wheel selection method.

        Parameters
        ----------
        population : list[numpy.ndarray]
            The population from which the parents will be selected.
        fitness : function
            The function that is used to evaluate the fitness of an individual.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing the selected parents.
        '''
        parents = random.choices(
            population=population,
            weights=[fitness(individual) for individual in population],
            k = 2)
        return parents[0], parents[1]



    def crossover(
            self, 
            parents
        ):
        '''Perform crossover operation on two parents to generate two offsprings.

        Parameters
        ----------
        parents : tuple of numpy.ndarray
            A tuple containing the two parents.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing the two offsprings generated from the parents.
        '''
        parent_a, parent_b = parents
        split_idx = random.randint(1, len(parent_a) - 1)
        offspring_x = np.concatenate(
            (parent_a[:split_idx], parent_b[split_idx:]),
            axis=None)
        offspring_y = np.concatenate(
            (parent_b[:split_idx], parent_a[split_idx:]),
            axis=None)
        return offspring_x, offspring_y



    def mutation(
            self,
            individual,
            probability : float
        ):
        '''Perform a mutation on an individual with a given probability.

        Parameters
        ----------
        individual : numpy.ndarray
            The individual to be mutated.
        probability : float
            The probability of mutation. Should be between 0 and 1.

        Returns
        -------
        numpy.ndarray
            The resulting individual after mutation.
        '''
        if random.random() <= probability:
            pos1 = random.randint(0, len(individual)-1)
            pos2 = random.randint(0, len(individual)-1)
            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        return individual



    def compute_next_generation(
        self,
        population, 
        n_elites: int,
        mutation_prob : float = 0.3, 
        ):
        '''Computes the next generation of a population using genetic algorithms.

        Parameters
        ----------
        population : list[numpy.ndarray]
            The current population of solutions.
        n_elites : int
            The number of elite solutions to maintain in the next generation.
        mutation_prob : loat, optional
            The probability of a mutation occurring, by default 0.3

        Returns
        -------
        list[numpy.ndarray]
            The next generation of solutions.
        '''
        next_generation = population[:n_elites]
        for _ in range(int((len(population) - n_elites)/2)):
            parents = self.roulette_selection(population, self.fitness)
            offspring = self.crossover(parents)
            next_generation += map(
                partial(self.mutation, probability = mutation_prob), offspring
            )
        return next_generation



    def early_stopping_check(
            self,
            current_best_fitness : float, 
            early_stopping_last_value : float, 
            early_stopping_count : int, 
            early_stopping_max: int
        ):
        """
        Check if the early stopping criteria has been met.
        
        Parameters
        ----------
        current_best_fitness : float
            The current best fitness of the population.
        early_stopping_last_value : float
            The last best fitness recorded.
        early_stopping_count : int
            The number of iterations where the best fitness has not improved.
        early_stopping_max : int
            The maximum number of iterations where the best fitness can not improve before stopping.
            
        Returns
        -------
        bool
            True if the early stopping criteria has been met, False otherwise.
        float
            Last best fitness recorded.
        int
            The number of iterations where the best fitness has not improved.
        """
        if current_best_fitness == early_stopping_last_value:
            early_stopping_count += 1
            if early_stopping_count == early_stopping_max:
                print(f'Early stopping happened, because best fitness did not change over {early_stopping_count} iterations')
                return True, early_stopping_last_value, early_stopping_count
        else:
            early_stopping_last_value = current_best_fitness
            early_stopping_count = 0
        return False, early_stopping_last_value, early_stopping_count



    def run_evolution(
            self,
            pop_size : int,
            individual_length : int,
            fitness_limit : float = 0.5,
            mutation_prob : float = 0.4,
            n_elites : int = 10,
            n_iter : int = 1000,
            early_stopping : bool = True,
            early_stopping_max : int = 10
        ):
        '''Runs the evolution process for a genetic algorithm.

        Parameters
        ----------
        pop_size : int
            The size of the population.
        individual_length : int
            The length of each individual in the population.
        fitness_limit : float, optional
            The minimum fitness value at which the evolution process will stop, by default 0.5
        mutation_prob : float, optional
            The probability of a mutation occurring, by default 0.4
        n_elites : int, optional
            The number of elite solutions to maintain in each generation, by default 10
        n_iter : int, optional
            The maximum number of iterations to run the evolution process, by default 1000
        early_stopping : bool, optional
            Indicates whether or not to use early stopping, by default True
        early_stopping_max : int, optional
            The maximum number of iterations where the best fitness can not improve before stopping, by default 10

        Returns
        -------
        numpy.ndarray
            The best individual of the final population.
        '''
        population = self.generate_population(pop_size, individual_length)
        early_stopping_last_value = -1
        early_stopping_count = 0
        pbar = tqdm(range(n_iter), 'Run evolution')
        for i in pbar:
            population = sorted(population, key=self.fitness)
            best_fitness = self.fitness(population[0])
            pbar.set_description(f'Generation: {i}, Best fitness: {round(best_fitness,4)}')

            if best_fitness <= fitness_limit:
                break

            if early_stopping:
                stop, early_stopping_last_value, early_stopping_count = self.early_stopping_check(best_fitness, 
                                                                                        early_stopping_last_value, 
                                                                                        early_stopping_count, 
                                                                                        early_stopping_max)
                if stop:
                    break

            population = self.compute_next_generation(population, n_elites, mutation_prob)

        return sorted(population, key = self.fitness)[0]