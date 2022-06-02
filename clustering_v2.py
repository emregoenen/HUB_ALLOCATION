from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from utils import ClusterHolder
from params import AffinityParams, MeanShiftParams, KMeansParams, SpectralParams, HierarchicalParams, DBSCANParams


class Clustering:
    def __init__(self, data):
        self.colorspace = np.array(
            ["purple", "cyan", "green", "orange", "brown", "gray", "magenta", "blue", "yellow", "pink"])
        self.data = data
        self.labels = None
        # self.center_points = None
        self.clustering = None
        self.get_params()
        self.get_clustering()
        self.find_labels()
        # self.get_central_points()
        self.cluster = ClusterHolder(self.data, self.labels)
        self.plot_clustering()

    def get_params(self):  # override this function
        ...

    def get_clustering(self):  # override this function
        ...

    def find_labels(self):
        self.labels = self.clustering.labels_
        print("\n##### CLUSTERING LABELS #####")
        print(self.labels)

    # def get_central_points(self):
    #     self.center_points = self.clustering.cluster_centers_
    #     print("\n##### CENTER POINTS #####")
    #     print(self.center_points)

    def plot_clustering(self):
        for index, i in enumerate(self.cluster.clusters):
            plt.scatter(i.center_point[0], i.center_point[1], color="red", marker="x", s=130)

        # cmap = get_cmap(len(self.data))
        # for i, (X,Y) in enumerate(self.data):
        #     plt.scatter(X, Y, color=cmap(i))
        #     plt.annotate(i, (X, Y))

        # for i, (X, Y) in enumerate(self.data):
        #     plt.scatter(X, Y, color=self.colorspace[i % len(self.colorspace)])
        #     plt.annotate(i, (X, Y))

        for i, (X, Y) in enumerate(self.data):
            plt.scatter(X, Y, color=self.colorspace[self.labels[i] % len(self.colorspace)])
            plt.annotate(i, (X, Y))

        for cluster in self.cluster.clusters:
            plt.scatter(cluster.central_node[0], cluster.central_node[1], color='red')
        plt.show()

    @staticmethod
    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n + 1)


class ClusterKMeans(Clustering):
    def __init__(self, data):
        self.params = KMeansParams()
        super().__init__(data)

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

    def get_clustering(self):
        self.clustering = KMeans(n_clusters=self.params.n_clusters, init=self.params.init, max_iter=self.params.max_iter, algorithm=self.params.algorithm).fit(self.data)
        print("\n##### KMEANS #####")
        print(self.clustering)


class ClusterAffinity(Clustering):
    def __init__(self, data):
        self.params = AffinityParams()
        super().__init__(data)

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

    def get_clustering(self):
        self.clustering = AffinityPropagation(damping=self.params.damping, max_iter=self.params.max_iter, convergence_iter=self.params.convergence_iter,
                             affinity=self.params.affinity, random_state=self.params.random_state).fit(self.data)
        print("\n##### AFFINITY PROPAGATION #####")
        print(self.clustering)


class ClusterMeanShift(Clustering):
    def __init__(self, data):
        self.params = MeanShiftParams()
        super().__init__(data)

    def get_params(self, bandwidth=None, max_iter=None, cluster_all=None):
        if bandwidth is not None:
            self.params.damping = bandwidth
        if max_iter is not None:
            self.params.max_iter = max_iter
        if cluster_all is not None:
            self.params.convergence_iter = cluster_all

        print("\n##### MEAN SHIFT PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = MeanShift(bandwidth=self.params.bandwidth, max_iter=self.params.max_iter, cluster_all=self.params.cluster_all).fit(self.data)
        print("\n##### MEAN SHIFT #####")
        print(self.clustering)


class ClusterSpectral(Clustering):
    def __init__(self, data):
        self.params = SpectralParams()
        super().__init__(data)

    def get_params(self, n_clusters=None, n_components=None, n_init=None, assign_labels=None):
        if n_clusters is not None:
            self.params.n_clusters = n_clusters
        if n_components is not None:
            self.params.n_components = n_components
        if n_init is not None:
            self.params.n_init = n_init
        if assign_labels is not None:
            self.params.assign_labels = assign_labels

        print("\n##### SPECTRAL CLUSTERING PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = SpectralClustering(n_clusters=self.params.n_clusters, n_components=self.params.n_components, n_init=self.params.n_init, assign_labels=self.params.assign_labels).fit(self.data)
        print("\n##### SPECTRAL CLUSTERING #####")
        print(self.clustering)


class ClusterHierarchical(Clustering):
    def __init__(self, data):
        self.params = HierarchicalParams()
        super().__init__(data)

    def get_params(self, n_clusters=None, affinity=None, linkage=None):
        if n_clusters is not None:
            self.params.n_clusters = n_clusters
        if affinity is not None:
            self.params.affinity = affinity
        if linkage is not None:
            self.params.linkage = linkage

        print("\n##### HIERARCHICAL CLUSTERING PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = AgglomerativeClustering(n_clusters=self.params.n_clusters, affinity=self.params.affinity, linkage=self.params.linkage).fit(self.data)
        print("\n##### HIERARCHICAL CLUSTERING #####")
        print(self.clustering)

class ClusterDBSCAN(Clustering):
    def __init__(self, data):
        self.params = DBSCANParams()
        super().__init__(data)

    def get_params(self, eps=None, min_samples=None, algorithm=None, p=None):
        if eps is not None:
            self.params.eps = eps
        if min_samples is not None:
            self.params.min_samples = min_samples
        if algorithm is not None:
            self.params.algorithm = algorithm
        if p is not None:
            self.params.p = p

        print("\n##### DBSCAN PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = DBSCAN(eps=self.params.eps, min_samples=self.params.min_samples, algorithm=self.params.algorithm, p=self.params.p).fit(self.data)
        print("\n##### DBSCAN CLUSTERING #####")
        print(self.clustering)

if __name__ == '__main__':

    # def get_data():  # --> np.array
    #     data = np.loadtxt("10.txt")
    #     print("\n##### DATA #####")
    #     print(data)
    #     return data

    my_data = np.loadtxt("100.txt")
    kmeans = ClusterKMeans(my_data)