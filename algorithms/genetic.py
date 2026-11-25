import random
import math
from typing import List

# Generate a random chromosome using permutation encoding.
# Each gene represents a unique item index, not its size.
# The chromosome determines the order in which items will
# be packed using the First Fit algorithm.
# This representation ensures that every item appears
# exactly once and supports Order Crossover (OX).
Individual = List[int]


def encode(items: List[int]) -> Individual:
    chromosome = list(range(len(items)))
    random.shuffle(chromosome)
    return chromosome


def decode(
    chromosome: Individual, items: List[int], bin_capacity: int
) -> List[List[int]]:
    # Decode a chromosome into bins using First Fit.
    # Each gene is an item index. Items are processed in
    # chromosome order and placed into the first bin with
    # enough remaining capacity.
    # A new bin is created when no existing bin can fit the item.
    if any(size > bin_capacity for size in items):
        raise ValueError("Item exceeds bin capacity")
    bins = []
    loads = []
    for item_index in chromosome:
        item_size = items[item_index]
        for i in range(len(bins)):
            if loads[i] + item_size <= bin_capacity:
                bins[i].append(item_index)
                loads[i] += item_size
                break

        else:
            bins.append([item_index])
            loads.append(item_size)

    return bins


#   Evaluate a chromosome using lexicographic fitness.
#     First priority: Minimize the number of bins.
#     Second priority: Maximize squared bin utilization.
#     Lower fitness values indicate better solutions.
def fitness(
    chromosome: Individual, items: List[int], bin_capacity: int
) -> tuple[int, float]:
    bins = decode(chromosome, items, bin_capacity)
    num_bins = len(bins)
    utilization_score = 0
    for bin_items in bins:
        bin_load = sum(items[index] for index in bin_items)
        utilization_score += (bin_load / bin_capacity) ** 2
    return (num_bins, -utilization_score)


# Generate the initial population for the Genetic Algorithm.
# Each individual is a randomly shuffled permutation
# of item indices, created using encode().
def initialize_population(items: List[int], population_size: int) -> List[Individual]:

    if type(population_size) is not int or population_size < 1:
        raise ValueError("Population size must be a positive integer")

    population = []
    for _ in range(population_size):
        population.append(encode(items))
    return population


# Randomly select chromosomes from the population.
# Evaluate the fitness of each selected chromosome.
# Choose the chromosome with the best (lowest) fitness.
# Return a copy of the winner to avoid modifying the original.
def tournament_selection(
    population: List[Individual],
    items: List[int],
    bin_capacity: int,
    tournament_size: int = 3,
) -> Individual:

    if not population:
        raise ValueError("Population cannot be empty")
    if tournament_size > len(population) or tournament_size < 1:
        raise ValueError("Invalid tournament size")

    candidates = random.sample(population, tournament_size)
    fitness_values = []
    for chromosome in candidates:
        fitness_values.append(fitness(chromosome, items, bin_capacity))
    best_index = fitness_values.index(min(fitness_values))
    winner = candidates[best_index]

    return winner.copy()


# Select two random crossover points.
# Copy a segment from the first parent to the child.
# Take the remaining genes from the second parent in the same order.
# Skip genes that already exist in the copied segment to avoid duplicates.
# Fill the empty positions in circular order.
# Repeat the process with swapped parents to create the second child.
def order_crossover(
    parent1: Individual, parent2: Individual
) -> tuple[Individual, Individual]:

    size = len(parent1)
    if size != len(parent2) or sorted(parent1) != sorted(parent2):
        raise ValueError("Parents must contain the same genes")
    if len(set(parent1)) != size:
        raise ValueError("Parents must be valid permutations")
    if size < 2:
        return parent1.copy(), parent2.copy()

    points = random.sample(range(size), 2)
    points.sort()
    start, end = points[0], points[1]

    def create_child(first_parent, second_parent):
        child = [None] * size
        child[start : end + 1] = first_parent[start : end + 1]
        used = set(child[start : end + 1])

        remaining = []
        for i in range(end + 1, end + 1 + size):
            gene = second_parent[i % size]
            if gene not in used:
                remaining.append(gene)

        position = (end + 1) % size
        for gene in remaining:
            while child[position] is not None:
                position = (position + 1) % size
            child[position] = gene
        return child

    child1 = create_child(parent1, parent2)
    child2 = create_child(parent2, parent1)
    return child1, child2


# Create a copy of the original chromosome.
# Check the mutation probability before applying mutation.
# Randomly select two different positions in the chromosome.
# Swap the genes at the selected positions.
# Return the modified chromosome while preserving a valid permutation.
def swap_mutation(chromosome: Individual, mutation_rate: float = 0.1) -> Individual:

    if mutation_rate < 0 or mutation_rate > 1:
        raise ValueError("Mutation rate must be between 0 and 1")

    mutated = chromosome.copy()
    if len(mutated) >= 2 and random.random() < mutation_rate:
        i = random.randint(0, len(mutated) - 1)
        j = random.randint(0, len(mutated) - 1)
        while i == j:
            j = random.randint(0, len(mutated) - 1)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


# Create a copy of the current population.
# Calculate the fitness of each chromosome.
# Sort chromosomes from best fitness to worst.
# Select the best chromosomes according to elite_size.
# Copy the selected chromosomes unchanged.
def apply_elitism(
    population: List[Individual],
    items: List[int],
    bin_capacity: int,
    elite_size: int = 2,
) -> List[Individual]:

    if elite_size < 0 or elite_size > len(population):
        raise ValueError("Invalid elite size")

    sorted_population = population.copy()
    #  sorted_population.sort(
    #     key=lambda chromosome: fitness(chromosome, items, bin_capacity)
    # )
    fitness_values = []
    for chromosome in sorted_population:
        fitness_values.append(fitness(chromosome, items, bin_capacity))

    for i in range(len(sorted_population)):
        for j in range(i + 1, len(sorted_population)):
            if fitness_values[i] > fitness_values[j]:
                fitness_values[i], fitness_values[j] = (
                    fitness_values[j],
                    fitness_values[i],
                )
                sorted_population[i], sorted_population[j] = (
                    sorted_population[j],
                    sorted_population[i],
                )
    elites = [chromosome.copy() for chromosome in sorted_population[:elite_size]]
    return elites

    # termination


def termination(
    generation: int,
    max_generations: int,
    best_fitness: tuple[int, float],
    items: List[int],
    bin_capacity: int,
    no_improvement: int,
    patience: int = 50,
) -> bool:
    if bin_capacity <= 0:
        raise ValueError("Bin capacity must be positive")
    if max_generations < 1:
        raise ValueError("Maximum generations must be positive")
    if patience < 1:
        raise ValueError("Patience must be positive")
    if generation < 0 or no_improvement < 0:
        raise ValueError("Counters cannot be negative")

    lower_bound = math.ceil(sum(items) / bin_capacity)

    if generation >= max_generations:
        return True

    if best_fitness[0] == lower_bound:
        return True

    if no_improvement >= patience:
        return True

    return False


def genetic_algorithm(
    items: List[int],
    bin_capacity: int,
    population_size: int = 100,
    max_generations: int = 200,
    tournament_size: int = 3,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elite_size: int = 2,
    patience: int = 50,
) -> tuple[Individual, List[List[int]], tuple[int, float]]:

    # 0. Input validation
    if not items:
        raise ValueError("Items cannot be empty")

    if type(bin_capacity) is not int or bin_capacity <= 0:
        raise ValueError("Bin capacity must be a positive integer")

    if any(type(size) is not int or size <= 0 for size in items):
        raise ValueError("Item sizes must be positive integers")

    if any(size > bin_capacity for size in items):
        raise ValueError("Item exceeds bin capacity")

    if type(population_size) is not int or population_size < 2:
        raise ValueError("Population size must be at least 2")

    if type(max_generations) is not int or max_generations < 1:
        raise ValueError("Maximum generations must be positive")

    if type(tournament_size) is not int or not 1 <= tournament_size <= population_size:
        raise ValueError("Invalid tournament size")

    if not 0 <= crossover_rate <= 1:
        raise ValueError("Crossover rate must be between 0 and 1")

    if not 0 <= mutation_rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1")

    if type(elite_size) is not int or not 0 <= elite_size < population_size:
        raise ValueError("Invalid elite size")

    if type(patience) is not int or patience < 1:
        raise ValueError("Patience must be positive")

    # 1. Initialize population
    population = initialize_population(items, population_size)

    # 2. Find the initial best chromosome
    best_chromosome = min(
        population, key=lambda chromosome: fitness(chromosome, items, bin_capacity)
    ).copy()

    best_fitness = fitness(best_chromosome, items, bin_capacity)

    generation = 0
    no_improvement = 0

    # 3. Repeat until a termination condition is reached
    while not termination(
        generation,
        max_generations,
        best_fitness,
        items,
        bin_capacity,
        no_improvement,
        patience,
    ):

        # 4. Preserve the best chromosomes
        new_population = apply_elitism(population, items, bin_capacity, elite_size)

        # 5. Generate offspring
        while len(new_population) < population_size:

            # Parent selection
            parent1 = tournament_selection(
                population, items, bin_capacity, tournament_size
            )

            parent2 = tournament_selection(
                population, items, bin_capacity, tournament_size
            )

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)

            # Add offspring to the new population
            new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        # 6. Replace the old population
        population = new_population

        # 7. Find the best chromosome in this generation
        current_best = min(
            population, key=lambda chromosome: fitness(chromosome, items, bin_capacity)
        )

        current_fitness = fitness(current_best, items, bin_capacity)

        # 8. Update the global best solution
        if current_fitness < best_fitness:
            best_chromosome = current_best.copy()
            best_fitness = current_fitness
            no_improvement = 0
        else:
            no_improvement += 1

        generation += 1

    # 9. Decode the best chromosome
    best_bins = decode(best_chromosome, items, bin_capacity)

    # 10. Return the final solution
    return best_chromosome, best_bins, best_fitness

    items = [4, 7, 2, 5, 8, 3, 6, 4]
    bin_capacity = 10

    best_chromosome, best_bins, best_fitness = genetic_algorithm(
        items=items,
        bin_capacity=bin_capacity,
        population_size=100,
        max_generations=200,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=2,
        patience=50,
    )

    print("Items:", items)
    print("Bin capacity:", bin_capacity)
    print("Best chromosome:", best_chromosome)
    print("Best fitness:", best_fitness)
    print("Number of bins:", len(best_bins))
    print("Theoretical minimum:", math.ceil(sum(items) / bin_capacity))

    print("\nBin Packing Result:")

    for i, bin_items in enumerate(best_bins):
        sizes = [items[index] for index in bin_items]
        load = sum(sizes)

        print(f"Bin {i + 1}: {sizes} | Load: {load}/{bin_capacity}")
