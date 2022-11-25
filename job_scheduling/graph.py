import json
import numpy as np


class Graph:

    def __init__(self):
        self.graph_dict = {}
        self.vmap = {}
        self.inv_vmap = {}
        self.same_nodes = {}
        self.__vertex_count = 0

    def add_vertex(self, vertex):
        label = vertex.split('_')[0]
        if label in self.same_nodes:
            self.same_nodes[label] += [self.__vertex_count]
        else:
            self.same_nodes[label] = [self.__vertex_count]

        self.graph_dict[vertex] = {}
        self.vmap[vertex] = self.__vertex_count
        self.inv_vmap[self.__vertex_count] = vertex
        self.__vertex_count += 1

    def add_edge(self, vertex1, vertex2, weight):
        self.graph_dict[vertex1][vertex2] = weight

    def generate_from_job(self, job):
        self.graph_dict = {}
        self.add_vertex('0')
        last_layer = ['0']
        for i in range(1, job['Deadlines'].max() + 1):
            curr = job[job['Deadlines'] >= i]
            this_layer = []
            for _, val in curr.iterrows():
                new_v = f'{val["Labels"]}_{i}'
                self.add_vertex(new_v)
                this_layer.append(new_v)
                for old_v in last_layer:
                    if old_v.split('_')[0] != val['Labels']:
                        self.add_edge(old_v, new_v, val['Profits'])

            last_layer = this_layer

    def to_adj_matrix(self):
        n = len(self.graph_dict)
        matrix = np.empty((n, n))
        matrix[:] = np.nan

        # i = 0
        for source_idx, source_val in self.graph_dict.items():
            for target_idx, target_val in source_val.items():
                matrix[self.vmap[source_idx]][self.vmap[target_idx]] = target_val

        return matrix

    def __str__(self):
        return json.dumps(self.graph_dict, indent=4).__str__()
