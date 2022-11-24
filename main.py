from graph import Graph
from ant_colony_optimizer import AntColonyOptimizer
import string

import pandas as pd
import numpy as np


def main():
    # job_labels = ['A', 'B', 'C', 'D', 'E']
    # job_deadlines = [2, 1, 2, 1, 3]
    # job_profits = [100, 25, 27, 19, 15]

    # job_labels = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    # job_deadlines = np.array([2, 1, 2, 1, 3, 4, 3, 5, 4, 1])
    # job_profits = np.array([100, 19, 27, 25, 15, 79, 23, 102, 30, 5])

    job_labels = np.array([* string.ascii_uppercase])
    job_deadlines = np.random.randint(1, 11, len(job_labels))
    job_profits = np.random.randint(1, 151, len(job_labels))

    job = pd.DataFrame({"Labels": job_labels,
                         "Deadlines": job_deadlines,
                         "Profits": job_profits})

    graph = Graph()
    graph.generate_from_job(job)
    aco = AntColonyOptimizer(graph, ants=20, evaporation_rate=0, intensification=0.1, tmax=100, alpha=0.5, beta=0.5)

    aco.fit()

    print(job_deadlines)
    print(job_profits)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
