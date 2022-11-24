import numpy as np


class AntColonyOptimizer:

    def __init__(self, graph, ants, evaporation_rate, intensification, alpha=1.0, beta=0.0,
                 beta_evaporation_rate=0, choose_best=0.1, tmax=100):
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.intensification = intensification
        self.alpha = alpha
        self.beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.choose_best = choose_best
        self.tmax = tmax

        # Internal representations
        self.graph = graph
        self.adj_matrix = graph.to_adj_matrix()
        self.pheromone_matrix = np.invert(np.isnan(self.adj_matrix)).astype(float)

        self.heuristic_matrix = self.pheromone_matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            self.heuristic_matrix = self.heuristic_matrix / self.heuristic_matrix.sum(axis=1)[:, None]
        self.heuristic_matrix[np.isnan(self.heuristic_matrix)] = 0.

        self.prob_matrix = None
        self._update_probabilities()

        # self.set_of_available_nodes = np.arange()

        # Internal stats
        self.best_series = []
        # self.best = None
        # self.fitted = False
        self.best_path = None
        self.best_score = -1
        # self.fit_time = None

        # TODO timing + plotting

        # Plotting values
        # self.stopped_early = False

    def _update_probabilities(self):
        self.prob_matrix = (self.pheromone_matrix ** self.alpha) * (self.heuristic_matrix ** self.beta)

    def _chose_next_node(self, curr_node, already_picked):
        # a = [self.graph.vmap[x] for x in already_picked]
        label = [x.split('_')[0] for x in already_picked]
        label_idx = [self.graph.same_nodes[x] for x in label]
        to_exclude = [x for y in label_idx for x in y]

        numerator = self.prob_matrix[self.graph.vmap[curr_node], :]
        for i in range(len(to_exclude)):
            numerator[to_exclude[i]] = 0.

        # print(numerator)
        if numerator.sum() == 0:
            return None

        # TODO choose_best ?
        denominator = np.sum(numerator)
        probabilities = numerator / denominator
        # print(probabilities)
        next_node_idx = np.random.choice(range(len(probabilities)), p=probabilities)
        return self.graph.inv_vmap[next_node_idx]

    def _path_score(self, path):
        score = 0
        coords_x = []
        coords_y = []
        for i in range(len(path) - 1):
            score += self.graph.graph_dict[path[i]][path[i + 1]]
            coords_x.append(self.graph.vmap[path[i]])
            coords_y.append(self.graph.vmap[path[i + 1]])

        return (coords_x, coords_y), score

    def _evaluate(self, paths):
        best_path_coords = None
        best_path = None
        best_score = -1
        for p in paths:
            coords, score = self._path_score(p)
            if score > best_score:
                best_path = p
                best_score = score
                best_path_coords = coords

        return best_path_coords, best_path, best_score

    def _evaporation(self):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        self.beta *= (1 - self.beta_evaporation_rate)

    def _intensify(self, coords):
        x = coords[0]
        y = coords[1]
        self.pheromone_matrix[x, y] += self.intensification

    # TODO prevent picking same node multiple times
    def fit(self):
        for t in range(self.tmax):
            paths = []
            for ant in range(self.ants):
                path = []
                curr_node = '0'

                while True:
                    path.append(curr_node)
                    # print(path)
                    if len(self.graph.graph_dict[curr_node]) != 0:
                        curr_node = self._chose_next_node(curr_node, path)
                        if curr_node is None:
                            break
                    else:
                        break

                paths.append(path)

            best_path_coords, best_path, best_score = self._evaluate(paths)
            # print(self._evaluate(paths))
            # TODO add early stopping

            if best_score > self.best_score:
                self.best_score = best_score
                self.best_path = best_path

            self.best_series.append(best_score)

            self._evaporation()
            # print(best_path_coords)
            self._intensify(best_path_coords)
            self._update_probabilities()

        # print(self.prob_matrix)
        # print(self.pheromone_matrix)
            print(f'Iteration {t + 1}/{self.tmax}: Best Score = {best_score}, Global Best Score = {self.best_score}')

        print(f'Best fit: Score = {self.best_score}, Path = {self.best_path}')