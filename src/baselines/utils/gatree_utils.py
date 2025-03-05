import numpy as np

MAX_ITER = int(1e5)


def tournament_selection(
    population: list,
    selection_tournament_size: int,
    fitness_key: str,
    reverse: bool,
    random: np.random.Generator,
):

    valid = False
    while not valid:
        selection = []

        # Select two trees
        for _ in range(2):
            indices = random.choice(
                len(population), selection_tournament_size, replace=False
            )
            candidates = [population[i] for i in indices]
            candidates.sort(key=lambda x: x.fitness[fitness_key], reverse=reverse)
            selection.append(candidates[0])

        # Check if trees are different
        if selection[0] != selection[1]:
            valid = True

    return selection[0], selection[1]
