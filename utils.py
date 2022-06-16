from itertools import combinations
import numpy as np

## Class to hold a cluster info
class Cluster:
    ## Constructor
    def __init__(self, index, points, center_point=None):
        self.central_node = None
        self.central_node_index = None
        self.distance_of_farthest_point = None
        self.cluster_index = index
        self.points = points
        if center_point is not None:
            self.center_point = center_point
        else:
            self.find_center_point()

        self.find_closest_point()
        self.find_distance_of_farthest_point()

    ## Check if given node is central node
    def is_central_node(self, point):
        if point == self.central_node:
            return True
        else:
            return False

    ## Find center point of cluster
    def find_center_point(self):
        center_point = np.sum(self.points, axis=0) / len(self.points)
        self.center_point = center_point

    ## Find closest point to the center point, which is central_node
    def find_closest_point(self): # Find central node (closest point to the center point)
        diff = (self.center_point - self.points)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        self.central_node_index = np.argmin(dist)
        self.central_node = self.points[self.central_node_index]

    ## Find distance of farthest point in the cluster to the cluster center node
    def find_distance_of_farthest_point(self):
        diff = (self.central_node - self.points)
        dist = max(np.sqrt(np.sum(diff ** 2, axis=-1)))
        self.distance_of_farthest_point = dist

    def __str__(self):
        return f"Cluster {self.cluster_index} =====> {self.points}, Center node = {self.central_node}"


## A class for holding and manipulating all clusters
class ClusterHolder:
    ## Constructor
    def __init__(self, data, labels, center_points=None):
        self.info = ""
        self.clusters = list()
        self.data = data
        self.labels = labels
        self.n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0) # DBSCAN returns -1 for noisy points, for DBSCAN we subtract 1 when noisy points are there
        if center_points is not None:
            self.center_points = center_points
        else:
            self.center_points = [None] * self.n_clusters

        self.split_into_clusters()

        self.objective_function = list()
        self.calculate_objective_function()

    ## Update initial solution info
    #  Call this method when cluster points move or changes
    def rewrite_info(self):
        self.info = ""
        self.print_splitted_clusters()
        self.print_cluster_center_nodes()
        self.calculate_objective_function()

    ## Split datacloud into clusters
    def split_into_clusters(self):
        for i in range(self.n_clusters):
            self.clusters.append(Cluster(index=i, points=self.data[np.where(self.labels == i)], center_point=self.center_points[i]))
        self.print_splitted_clusters()
        self.print_cluster_center_nodes()

    ## Generate splitted cluster info
    def print_splitted_clusters(self):
        self.info += f"\nThere are {len(self.clusters)} clusters\n"
        for i, cluster in enumerate(self.clusters):
            strng = "" + "Cluster " + str(i) + " ======>"
            temp = []
            for point in cluster.points:
                temp.append(self.get_data_index_by_point(point))
            strng += " " + str(temp)
            self.info += strng + '\n'

    ## Generate clusters' center node info
    def print_cluster_center_nodes(self):
        self.info += "Cluster center nodes ======> "
        temp = []
        for cluster in self.clusters:
            temp.append(self.get_data_index_by_point(cluster.central_node))
        self.info += str(temp) + '\n'

    ## Calculate the objective function
    def calculate_objective_function(self):
        self.info += "\n------Farthest hub distances------\n"
        self.objective_function = list()
        hub_dist = dict()
        for cluster in self.clusters:
            cluster.find_center_point()
            cluster.find_distance_of_farthest_point()
            hub_dist.setdefault(self.get_data_index_by_point(cluster.central_node), cluster.distance_of_farthest_point)
        self.info += str(hub_dist) + '\n'

        pair_list = list(combinations(range(self.n_clusters),2))
        self.info += "\nAll possible pairs : \n"
        possible_pairs = list()
        for pair in pair_list:
            possible_pairs.append([self.get_data_index_by_point(self.get_cluster_by_index(pair[0]).central_node), self.get_data_index_by_point(self.get_cluster_by_index(pair[1]).central_node)])
        self.info += str(pair_list) + '\n'

        for pair in pair_list:
            self.objective_function.append(self.get_cluster_by_index(pair[0]).distance_of_farthest_point + 0.75 * self.find_distance_between_clusters(pair[0], pair[1]) + self.get_cluster_by_index(pair[0]).distance_of_farthest_point)
        self.objective_function.append(2*max(self.get_cluster_distance_of_farthest_points()))

        self.info += "\n------Pair objectives------\n"
        self.info += str(self.objective_function)
        self.info += f"\nObjective function ======> {max(self.objective_function)}\n"

        return max(self.objective_function)

    ## Returns index of the point in data cloud
    def get_data_index_by_point(self, point):
        for indx, dt in enumerate(self.data):
            if np.array_equal(dt, point):
                return indx
        return -1

    ## Calculate and return distances of farthest point for each cluster
    def get_cluster_distance_of_farthest_points(self):
        distance_of_farthest_points = list()
        for cluster in self.clusters:
            distance_of_farthest_points.append(cluster.distance_of_farthest_point)
        return distance_of_farthest_points

    ## Calculates and returns distance between two cluster
    # @param a ClusterA
    # @param b ClusterB
    def find_distance_between_clusters(self, a, b): # Finds the distance between central nodes of cluster_a and cluster_b
        diff = self.get_cluster_by_index(a).central_node - self.get_cluster_by_index(b).central_node
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        return dist

    ## Returns cluster by given index
    def get_cluster_by_index(self, index):
        for cluster in self.clusters:
            if cluster.cluster_index == index:
                return cluster
        return None