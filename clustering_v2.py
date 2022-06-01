from sklearn.cluster import KMeans, AffinityPropagation
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from itertools import combinations


@dataclass
class KMeansParams:
    n_clusters: int = 3
    init: str = "k-means++"
    max_iter: int = 300
    algorithm: str = "auto"


@dataclass
class AffinityParams:
    damping: float = 0.5
    max_iter: int = 200
    convergence_iter: int = 15
    affinity: str = 'euclidean'
    random_state: int = None


class Clustering:

    def __init__(self):
        self.colorspace = np.array(
            ["orange", "purple", "brown", "gray", "cyan", "magenta", "green", "blue", "yellow", "pink"])
        self.plot = plt

    def get_params(self):  # override this function
        ...


class Cluster:
    def __init__(self, index, points, center_point=None):
        self.central_node = None
        self.distance_of_farthest_point = None
        self.cluster_index = index
        self.points = points
        if center_point is not None:
            self.center_point = center_point
        else:
            self.center_point = self.find_center_point()

        self.find_closest_point()
        self.find_distance_of_farthest_point()

    def find_center_point(self):
        center_point = np.sum(self.points, axis=0) / len(self.points)
        return center_point

    def find_closest_point(self): # Find central node (closest point to the center point)
        diff = (self.center_point - self.points)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        self.central_node = self.points[np.argmin(dist)]

    def find_distance_of_farthest_point(self): # Find distance of farthest point in the cluster to the cluster center node
        diff = (self.central_node - self.points)
        dist = max(np.sqrt(np.sum(diff ** 2, axis=-1)))
        self.distance_of_farthest_point = dist

    def __str__(self):
        return f"Cluster {self.cluster_index} =====> {self.points}, Center point = {self.center_point}"


class ClusterHolder:
    def __init__(self, data, labels, center_points=None):
        self.clusters = list()
        self.data = data
        self.labels = labels
        self.n_clusters = len(np.unique(labels))
        if center_points is not None:
            self.center_points = center_points
        else:
            self.center_points = [None] * self.n_clusters

        self.split_into_clusters()

        self.objective_function = list()
        self.calculate_objective_function()

    def split_into_clusters(self):
        for i in range(self.n_clusters):
            self.clusters.append(Cluster(index=i, points=self.data[np.where(self.labels == i)], center_point=self.center_points[i]))
        self.print_splitted_clusters()

    def print_splitted_clusters(self):
        print("\n##### SPLITTED CLUSTERS #####")
        for cluster in self.clusters:
            print(cluster)

    def calculate_objective_function(self):
        print("\n##### FARTHEST HUB DISTANCES #####")
        hub_dist = dict()
        for cluster in self.clusters:
            hub_dist.setdefault(self.get_data_index_by_point(cluster.central_node), cluster.distance_of_farthest_point)
        print(hub_dist)

        pair_list = list(combinations(range(self.n_clusters),2))
        print("\n##### ALL POSSIBLE PAIRS #####")
        possible_pairs = list()
        for pair in pair_list:
            possible_pairs.append([self.get_data_index_by_point(self.get_cluster_by_index(pair[0]).central_node), self.get_data_index_by_point(self.get_cluster_by_index(pair[1]).central_node)])
        print(possible_pairs)

        for pair in pair_list:
            self.objective_function.append(self.get_cluster_by_index(pair[0]).distance_of_farthest_point + 0.75 * self.find_distance_between_clusters(pair[0], pair[1]) + self.get_cluster_by_index(pair[0]).distance_of_farthest_point)
        self.objective_function.append(2*max(self.get_cluster_distance_of_farthest_points()))

        print("\n##### PAIR OBJECTIVES #####")
        print(self.objective_function)
        print("\n##### OBJECTIVE FUNCTION #####")
        print(max(self.objective_function))

    def get_data_index_by_point(self, point):
        for indx, dt in enumerate(self.data):
            if np.array_equal(dt, point):
                return indx
        return -1

    def get_cluster_distance_of_farthest_points(self):
        distance_of_farthest_points = list()
        for cluster in self.clusters:
            distance_of_farthest_points.append(cluster.distance_of_farthest_point)
        return distance_of_farthest_points

    def find_distance_between_clusters(self, a, b): # Finds the distance between central nodes of cluster_a and cluster_b
        diff = self.get_cluster_by_index(a).central_node - self.get_cluster_by_index(b).central_node
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        return dist

    def get_cluster_by_index(self, index):
        for cluster in self.clusters:
            if cluster.cluster_index == index:
                return cluster
        return None

class ClusterKMeans(Clustering):
    def __init__(self):
        super().__init__()
        self.data = None
        self.kmeans = None
        self.labels = None
        self.center_points = None
        self.params = KMeansParams()

        self.get_data()
        self.get_params()
        self.get_kmeans()
        self.find_labels()
        self.get_central_points()
        self.cluster = ClusterHolder(self.data, self.labels, self.params.n_clusters)

        self.plot_clustering()

    def find_labels(self):
        self.labels = self.kmeans.labels_
        print("\n##### CLUSTERING LABELS #####")
        print(self.labels)

    def get_central_points(self):
        self.center_points = self.kmeans.cluster_centers_
        print("\n##### CENTER POINTS #####")
        print(self.center_points)

    def get_params(self, n_clusters=None, init=None, max_iter=None, algorithm=None):
        if n_clusters is not None:
            self.params.n_clusters = n_clusters
        if init is not None:
            self.params.init = init
        if max_iter is not None:
            self.params.max_iter = max_iter
        if algorithm is not None:
            self.params.algorithm = algorithm

        print("\n##### K-MEANS PARAMETERS #####")
        print(self.params)

    def get_data(self): # --> np.array
        self.data = np.loadtxt("10.txt")
        print("\n##### DATA #####")
        print(self.data)

    def get_kmeans(self):
        self.kmeans = KMeans(n_clusters=self.params.n_clusters, init=self.params.init, max_iter=self.params.max_iter, algorithm=self.params.algorithm).fit(self.data)
        print("\n##### KMEANS #####")
        print(self.kmeans)

    def plot_clustering(self):
        for index, i in enumerate(self.cluster.clusters):
            plt.scatter(i.center_point[0], i.center_point[1], color="red", marker="x", s=130)

        # cmap = get_cmap(len(self.data))
        # for i, (X,Y) in enumerate(self.data):
        #     plt.scatter(X, Y, color=cmap(i))
        #     plt.annotate(i, (X, Y))

        for i,(X,Y) in enumerate(self.data):
            plt.scatter(X, Y, color=self.colorspace[i % len(self.colorspace)])
            plt.annotate(i, (X, Y))

        for cluster in self.cluster.clusters:
            plt.scatter(cluster.central_node[0], cluster.central_node[1], color='red')
        plt.show()


class ClusterAffinity(Clustering):
    def __init__(self):
        super().__init__()
        self.data = None
        self.clustering = None
        self.labels = None
        self.center_points = None
        self.params = AffinityParams()

        self.get_data()
        self.get_params()
        self.get_clustering()
        self.find_labels()
        self.get_central_points()
        self.cluster = ClusterHolder(self.data, self.labels)

        self.plot_clustering()

    def find_labels(self):
        self.labels = self.clustering.labels_
        print("\n##### CLUSTERING LABELS #####")
        print(self.labels)

    def get_central_points(self):
        self.center_points = self.clustering.cluster_centers_
        print("\n##### CENTER POINTS #####")
        print(self.center_points)

    def get_params(self, damping=None, max_iter=None, convergence_iter=None, affinity=None, random_state=None):
        if damping is not None:
            self.params.damping = damping
        if max_iter is not None:
            self.params.max_iter = max_iter
        if convergence_iter is not None:
            self.params.convergence_iter = convergence_iter
        if affinity is not None:
            self.params.affinity = affinity
        if random_state is not None:
            self.params.random_state = random_state

        print("\n##### AFFINITY PROPAGATION PARAMETERS #####")
        print(self.params)

    def get_data(self):  # --> np.array
        self.data = np.loadtxt("10.txt")
        print("\n##### DATA #####")
        print(self.data)

    def get_clustering(self):
        self.clustering = AffinityPropagation(damping=self.params.damping, max_iter=self.params.max_iter, convergence_iter=self.params.convergence_iter,
                             affinity=self.params.affinity, random_state=self.params.random_state).fit(self.data)
        print("\n##### KMEANS #####")
        print(self.clustering)

    def plot_clustering(self):
        for index, i in enumerate(self.cluster.clusters):
            plt.scatter(i.center_point[0], i.center_point[1], color="red", marker="x", s=130)

        # cmap = get_cmap(len(self.data))
        # for i, (X,Y) in enumerate(self.data):
        #     plt.scatter(X, Y, color=cmap(i))
        #     plt.annotate(i, (X, Y))

        for i, (X, Y) in enumerate(self.data):
            plt.scatter(X, Y, color=self.colorspace[i % len(self.colorspace)])
            plt.annotate(i, (X, Y))

        for cluster in self.cluster.clusters:
            plt.scatter(cluster.central_node[0], cluster.central_node[1], color='red')
        plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n+1)

if __name__ == '__main__':
    kmeans = ClusterAffinity()