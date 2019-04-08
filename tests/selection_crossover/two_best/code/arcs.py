from PIL import Image, ImageChops, ImageDraw, ImageFont
from skimage.measure import compare_mse
import numpy as np
from math import sqrt
import copy

NUMBER_OF_GENERATIONS = 35000

POPULATION_NUMBER = 15
MUTATION_RATE = 0.7

LETTERS = [chr(i) for i in range(97, 123)]

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


def crossover(population, fit_values):
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

    for i in range(POPULATION_NUMBER):
        if i != best_fit_idx and i != second_best_fit_idx:
            population[i] = ImageChops.blend(population[i], population[best_fit_idx], 0.5)


def mutation(population):
    indicators = np.random.random(POPULATION_NUMBER) < MUTATION_RATE
    mutation_number = np.sum(indicators)
    colors = np.random.randint(0, 256, (mutation_number, 3), dtype=np.uint8)

    initial_location = np.random.random_integers(-5, max(img_size), (mutation_number, 2))
    lengths = np.random.randint(1, 150, (mutation_number, 2))

    degrees = np.random.random_integers(-180, 180, mutation_number)

    count = 0
    for i in range(POPULATION_NUMBER):
        if indicators[i]:
            draw = ImageDraw.Draw(population[i])
            draw.arc([tuple(initial_location[count]),
                      tuple(initial_location[count] + lengths[count])],
                     degrees[count],
                     np.random.random_integers(degrees[count], degrees[count] + 60),
                     fill=tuple(colors[count]), width=1)
            del draw
            count += 1


def compare_populations(prev_fit_values, new_fit_values):
    if np.sum(prev_fit_values) < np.sum(new_fit_values):
        return 0
    else:
        return 1


def main():
    population = initialization(POPULATION_NUMBER)

    fit_values = np.empty(POPULATION_NUMBER)
    new_fit_values = np.empty(POPULATION_NUMBER)

    for i in range(POPULATION_NUMBER):
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

            out_file = open("../results/arcs/mona_lisa_" + str(generation) + ".jpg", 'wb')
            best_solution.save(out_file, "JPEG")
            out_file.flush()
            out_file.close()

            for i in range(POPULATION_NUMBER):
                np.save("../populations/arcs/population_mona_lisa_" + str(i), np.asarray(population[i]))

        new_population = copy.deepcopy(population)

        crossover(new_population, fit_values)

        mutation(new_population)

        for i in range(POPULATION_NUMBER):
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