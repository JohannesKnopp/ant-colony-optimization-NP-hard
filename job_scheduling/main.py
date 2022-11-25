from graph import Graph
from ant_colony_optimizer import AntColonyOptimizer
import string
import uuid

import pandas as pd
import numpy as np


def main():
    # job_labels = ['A', 'B', 'C', 'D', 'E']
    # job_deadlines = [2, 1, 2, 1, 3]
    # job_profits = [100, 25, 27, 19, 15]

    # job_labels = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    # job_deadlines = np.array([2, 1, 2, 1, 3, 4, 3, 5, 4, 1])
    # job_profits = np.array([100, 19, 27, 25, 15, 79, 23, 102, 30, 5])

    # job_labels = np.array([uuid.uuid4() for _ in range(100)])
    # job_deadlines = np.random.randint(1, 100, len(job_labels))
    # job_profits = np.random.randint(1, 401, len(job_labels))

    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    with open("labels.npy", "rb") as f:
        job_labels = np.load(f)

    with open("deads.npy", "rb") as f:
        job_deadlines = np.load(f)

    with open("profit.npy", "rb") as f:
        job_profits = np.load(f)


    # job_labels = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
    #                        "R", "S", "T", "U", "V","W", "X", "Y", "Z"])
    # job_deadlines = np.array([2, 1, 2, 1, 3, 4, 3, 5, 4, 1, 5, 4, 3, 1, 2, 6, 4, 5, 1, 6, 3, 3, 1, 2, 5, 6])
    # job_profits = np.array([100, 19, 27, 25, 15, 79, 23, 102, 30, 5, 12, 33, 25, 12, 35, 65, 66, 23,
    #                         78, 6, 28, 5, 24, 65, 77, 4])

    job = pd.DataFrame({"Labels": job_labels,
                         "Deadlines": job_deadlines,
                         "Profits": job_profits})

    graph = Graph()
    graph.generate_from_job(job)
    # aco = AntColonyOptimizer(graph, ants=100, evaporation_rate=0.3, intensification=0.5, tmax=100, alpha=0.3, beta=1) -> 501 convergence
    aco = AntColonyOptimizer(graph, ants=5, evaporation_rate=0.3, intensification=0.5, tmax=1000, alpha=0.2, beta=1)

    aco.fit()

    print(job_deadlines)
    print(job_profits)
    # print(graph)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
