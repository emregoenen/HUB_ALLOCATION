from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from utils import ClusterHolder
from params import AffinityParams, MeanShiftParams, KMeansParams, SpectralParams, HierarchicalParams, DBSCANParams


class Clustering:
    def __init__(self, data, params, ui, driver):
        self.colorspace = np.array(
            ["purple", "cyan", "green", "orange", "brown", "gray", "magenta", "blue", "yellow", "pink"])
        self.params = params
        self.data = data
        self.labels = None
        self.clustering = None
        self.ui = ui
        self.driver = driver
        self.print_params()
        self.get_clustering()
        self.find_labels()
        self.cluster = ClusterHolder(self.data, self.labels)
        self.plot_clustering()
        self.__post_init__()

    def __post_init__(self):
        self.driver.check_buttons()
        self.driver.update_input_image()

    def get_cluster_holder(self):
        return self.cluster

    def get_clustering(self):  # override this function
        ...

    def print_params(self): # override this function
        ...

    def find_labels(self):
        self.labels = self.clustering.labels_
        self.driver.print_info("\nClustering labels")
        self.driver.print_info(self.labels)

    def plot_clustering(self):
        plt.clf()
        for index, i in enumerate(self.cluster.clusters):
            plt.scatter(i.center_point[0], i.center_point[1], color="red", marker="x", s=130)

        for i, (X, Y) in enumerate(self.data):
            plt.scatter(X, Y, color=self.colorspace[self.labels[i] % len(self.colorspace)])
            plt.annotate(i, (X, Y))

        for cluster in self.cluster.clusters:
            plt.scatter(cluster.central_node[0], cluster.central_node[1], color='red')


        plt.savefig("resources/temp/input.png")
        # plt.show()


class ClusterKMeans(Clustering):
    def __init__(self, data, params, ui, driver):
        super().__init__(data, params, ui, driver)

    def print_params(self):
        self.driver.print_info("\nK-means parameters")
        self.driver.print_info(self.params)

    def get_clustering(self):
        self.clustering = KMeans(n_clusters=self.params.n_clusters, init=self.params.init, max_iter=self.params.max_iter, algorithm=self.params.algorithm).fit(self.data)


class ClusterAffinity(Clustering):
    def __init__(self, data, params, ui, driver):
        super().__init__(data, params, ui, driver)

    def print_params(self):
        self.driver.print_info("\nAffinity propagation parameters")
        self.driver.print_info(self.params)

    def get_clustering(self):
        self.clustering = AffinityPropagation(damping=self.params.damping, max_iter=self.params.max_iter, convergence_iter=self.params.convergence_iter,
                             affinity=self.params.affinity, random_state=self.params.random_state).fit(self.data)


class ClusterMeanShift(Clustering):
    def __init__(self, data, params, ui, driver):
        super().__init__(data, params, ui, driver)

    def print_params(self):
        self.driver.print_info("\nMean shift parameters")
        self.driver.print_info(self.params)

    def get_clustering(self):
        self.clustering = MeanShift(bandwidth=self.params.bandwidth, max_iter=self.params.max_iter, cluster_all=self.params.cluster_all).fit(self.data)


class ClusterSpectral(Clustering):
    def __init__(self, data, params, ui, driver):
        super().__init__(data, params, ui, driver)

    def print_params(self):
        self.driver.print_info("\nSpectral clustering parameters")
        self.driver.print_info(self.params)

    def get_clustering(self):
        self.clustering = SpectralClustering(n_clusters=self.params.n_clusters, n_components=self.params.n_components, n_init=self.params.n_init, assign_labels=self.params.assign_labels).fit(self.data)


class ClusterHierarchical(Clustering):
    def __init__(self, data, params, ui, driver):
        super().__init__(data, params, ui, driver)

    def print_params(self):
        self.driver.print_info("\nHierarchical clustering parameters")
        self.driver.print_info(self.params)

    def get_clustering(self):
        self.clustering = AgglomerativeClustering(n_clusters=self.params.n_clusters, affinity=self.params.affinity, linkage=self.params.linkage).fit(self.data)


class ClusterDBSCAN(Clustering):
    def __init__(self, data, params, ui, driver):
        super().__init__(data, params, ui, driver)

    def print_params(self):
        self.driver.print_info("\nDbscan clustering parameters")
        self.driver.print_info(self.params)

    def get_clustering(self):
        self.clustering = DBSCAN(eps=self.params.eps, min_samples=self.params.min_samples, algorithm=self.params.algorithm, p=self.params.p).fit(self.data)