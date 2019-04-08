from PIL import Image, ImageChops, ImageDraw
from skimage.measure import compare_mse
import numpy as np
from math import sqrt
import copy

# Number of iterations the algorithm works
NUMBER_OF_GENERATIONS = 40000

# Number of chromosomes into the population
POPULATION_NUMBER = 15

# Probability of mutation
MUTATION_RATE = 0.7

# Initial images into population
def initialization(population_number: int):
    # return just completely black images
    return [Image.new('RGB', img_size, (0, 0, 0)) for i in range(population_number)]

# Fitness function based on root mean square aglorithm
def fitness_function_rms(chromosome):
    histogram = np.asarray(ImageChops.difference(chromosome, img).histogram())
    sq = (value * ((idx % 256) ** 2) for idx, value in enumerate(histogram))
    sum_of_squares = sum(sq)
    rms = sqrt(sum_of_squares / float(img_size[0] * img_size[1]))
    return rms

# Fitness function based on mean square error algorithm. Uses library functionality
def fitness_function_mse(chromosome):
    return compare_mse(np.asarray(chromosome), np.asarray(img))

# Crossover function
def crossover(population, fit_values):
    # searches for two best chromosomes to leave them unchanged
    if fit_values[0] < fit_values[1]:
        best_fit_idx = 0
        best_fit = fit_values[0]
        second_best_fit_idx = 1
        second_best_fit = fit_values[1]
    else:
        best_fit_idx = 1
        best_fit = fit_values[1]
        second_best_fit_idx = 0
        second_best_fit = fit_values[0]

    for i in range(2, POPULATION_NUMBER):
        if fit_values[i] < best_fit:
            second_best_fit = best_fit
            second_best_fit_idx = best_fit_idx
            best_fit = fit_values[i]
            best_fit_idx = i
        elif fit_values[i] < second_best_fit:
            second_best_fit = fit_values[i]
            second_best_fit_idx = i

    # combines all except the two best chromosomes with the best one
    for i in range(POPULATION_NUMBER):
        if i != best_fit_idx and i != second_best_fit_idx:
            population[i] = ImageChops.blend(population[i], population[best_fit_idx], 0.5)

# Mutation function
def mutation(population):
    # choice of chromosomes that will mutatate
    indicators = np.random.random(POPULATION_NUMBER) < MUTATION_RATE
    mutation_number = np.sum(indicators)

    # generation of random parameters for object which will be added into the image
    colors= np.random.random_integers(0, 255, mutation_number)                              # color shade of the arc
    initial_location = np.random.random_integers(-5, max(img_size), (mutation_number, 2))   # initial location of the arc
    lengths = np.random.randint(1, 150, (mutation_number, 2))                               # length of the arc
    degrees = np.random.random_integers(-180, 180, mutation_number)                         # curve degree in angles of the arc

    # mutation applying
    count = 0
    for i in range(POPULATION_NUMBER):
        if indicators[i]:   # if chromosome should mutate
            # draws specified object with generated parameters
            draw = ImageDraw.Draw(population[i])
            # draws arc with mono color
            draw.arc([tuple(initial_location[count]),
                      tuple(initial_location[count] + lengths[count])],
                     degrees[count],
                     np.random.random_integers(degrees[count], degrees[count] + 50),
                     fill=(colors[count], colors[count], colors[count]), width=1)
            del draw
            count += 1

# compare two populations as a sum of fitness function applyed to each chromosome into the population
def compare_populations(prev_fit_values, new_fit_values):
    if np.sum(prev_fit_values) < np.sum(new_fit_values):
        return 0
    else:
        return 1


def main():
    # population initialization
    population = initialization(POPULATION_NUMBER)

    # declaration of arrays which will keep fitness values for chromosomes into populations
    fit_values = np.empty(POPULATION_NUMBER)
    new_fit_values = np.empty(POPULATION_NUMBER)

    # initialization of fitness values for initial population
    for i in range(POPULATION_NUMBER):
        fit_values[i] = fitness_function_rms(population[i])

    # one loop iteration processes one generation
    generation = 0
    while generation < NUMBER_OF_GENERATIONS:
        # in order to follow how many generations have been processed
        if (generation % 100) == 0:
            print("Generation ", generation, sep=" ")

        # save resultant image and images into the current population
        if (generation % 500) == 0:

            # choose the best chromosome to save as image
            best_solution = population[0]
            best_solution_index = 0
            best_fitness = fit_values[0]
            for i in range(1, len(population)):
                current_fitness = fit_values[i]
                if current_fitness < best_fitness:
                    best_solution = population[i]
                    best_solution_index = i
                    best_fitness = current_fitness

            print(fit_values[best_solution_index])

            out_file = open("../results/output_arcs_engraving.jpg", 'wb')
            best_solution.save(out_file, "JPEG")
            out_file.flush()
            out_file.close()

            # backup all chromosomes in current population
            for i in range(POPULATION_NUMBER):
                np.save("../populations/population_output_arcs_engraving_" + str(i), np.asarray(population[i]))

        # create new array of images for the new population
        new_population = copy.deepcopy(population)

        # apply crossover operation
        crossover(new_population, fit_values)

        # apply mutation
        mutation(new_population)

        # compute fitness values for the new population
        for i in range(POPULATION_NUMBER):
            new_fit_values[i] = fitness_function_rms(new_population[i])

        # choose which population to save in the next generation
        if compare_populations(fit_values, new_fit_values):
            population = new_population
            fit_values, new_fit_values = new_fit_values, fit_values

        generation += 1

# open input image
img = Image.open("../images/input.jpg")
img_size = img.size

if __name__ == '__main__':
    main()