from PIL import Image, ImageChops, ImageDraw, ImageFont
from skimage.measure import compare_mse
import numpy as np
from math import sqrt

NUMBER_OF_GENERATIONS = 35000

POPULATION_SIZE = 15
MUTATION_RATE = 0.7
CROSSOVER_RATE = 0.6

def initialization(population_number: int):
    return [Image.new('RGB', img_size, (0, 0, 0)) for i in range(population_number)]


def fitness_function_rms(chromosome):
    histogram = np.asarray(ImageChops.difference(chromosome, img).histogram())
    sq = (value * ((idx % 256) ** 2) for idx, value in enumerate(histogram))
    sum_of_squares = sum(sq)
    rms = sqrt(sum_of_squares / float(img_size[0] * img_size[1]))
    return rms

def fitness_function_mse(chromosome):
    return compare_mse(np.asarray(chromosome), np.asarray(img))


def roulette_wheel(population, fit_values):
    fitness = []
    total = 0
    for i in range(POPULATION_SIZE):
        fitness.append(1 / (1 + fit_values[i]))
        total += fitness[i]

    probability = []
    for i in range(POPULATION_SIZE):
        probability.append(fitness[i] / total)

    cumulative_probability = []
    cumulative_sum = 0
    for i in range(POPULATION_SIZE):
        cumulative_probability.append(cumulative_sum + probability[i])
        cumulative_sum += probability[i]

    new_population = []
    new_fit_values = []
    r_values = np.random.random(POPULATION_SIZE)
    for i in range(POPULATION_SIZE):
        j = 0
        while j <= POPULATION_SIZE and r_values[i] > cumulative_probability[j]:
            j += 1
        new_population.append(population[j])
        new_fit_values.append(fit_values[j])

    return new_population, new_fit_values

def crossover(population):
    parent_indexes = []
    r_values = np.random.random(POPULATION_SIZE)
    for i in range(POPULATION_SIZE):
        if r_values[i] < CROSSOVER_RATE:
            parent_indexes.append(i)

    if (len(parent_indexes)) > 1:
        for i in range(len(parent_indexes) - 1):
            population[parent_indexes[i]] = ImageChops.blend(population[parent_indexes[i]],
                                                             population[parent_indexes[i + 1]], 0.5)

        population[parent_indexes[-1]] = ImageChops.blend(population[parent_indexes[-1]],
                                                          population[parent_indexes[1]], 0.5)

def mutation(population):
    indicators = np.random.random(POPULATION_SIZE) < MUTATION_RATE
    mutation_number = np.sum(indicators)
    colors = np.random.randint(0, 256, (mutation_number, 3), dtype=np.uint8)

    initial_points = np.random.random_integers(-5, max(img_size), (mutation_number, 2))
    triangle_sizes = np.random.random_integers(20, 70, (mutation_number, 2, 2))

    count = 0
    for i in range(POPULATION_SIZE):
        if indicators[i]:
            draw = ImageDraw.Draw(population[i])
            draw.polygon([tuple(initial_points[count]),
                          tuple(initial_points[count] + triangle_sizes[count][0]),
                          tuple(initial_points[count] + triangle_sizes[count][1])],
                         fill=tuple(colors[count]))
            del draw
            count += 1


def compare_populations(prev_fit_values, new_fit_values):
    if np.sum(prev_fit_values) < np.sum(new_fit_values):
        return 0
    else:
        return 1


def main():
    population = initialization(POPULATION_SIZE)

    fit_values = np.empty(POPULATION_SIZE)

    for i in range(POPULATION_SIZE):
        fit_values[i] = fitness_function_rms(population[i])

    generation = 0
    while generation < NUMBER_OF_GENERATIONS:
        if (generation % 100) == 0:
            print("Generation ", generation, sep=" ")

        if (generation % 500) == 0:

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

            out_file = open("../results/triangles/mona_lisa_" + str(generation) + ".jpg", 'wb')
            best_solution.save(out_file, "JPEG")
            out_file.flush()
            out_file.close()

            for i in range(POPULATION_SIZE):
                np.save("../populations/triangles/population_mona_lisa_" + str(i), np.asarray(population[i]))

        new_population, new_fit_values = roulette_wheel(population, fit_values)

        crossover(new_population)

        mutation(new_population)

        for i in range(POPULATION_SIZE):
            new_fit_values[i] = fitness_function_rms(new_population[i])

        if compare_populations(fit_values, new_fit_values):
            population = new_population
            fit_values, new_fit_values = new_fit_values, fit_values

        generation += 1

font = ImageFont.truetype("../../../../fonts/Dokdo-Regular.ttf", 50)
img = Image.open("../../../../images/mona_lisa.jpg")
img_size = img.size

if __name__ == '__main__':
    main()