from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from utils import ClusterHolder
from params import AffinityParams, MeanShiftParams, KMeansParams, SpectralParams, HierarchicalParams, DBSCANParams


class Clustering:
    def __init__(self, data, params, ui):
        self.colorspace = np.array(
            ["purple", "cyan", "green", "orange", "brown", "gray", "magenta", "blue", "yellow", "pink"])
        self.params = params
        self.data = data
        self.labels = None
        self.clustering = None
        self.ui = ui
        self.print_params()
        self.get_clustering()
        self.find_labels()
        self.cluster = ClusterHolder(self.data, self.labels, self.ui)
        self.plot_clustering()


    def get_clustering(self):  # override this function
        ...

    def print_params(self): # override this function
        ...

    def find_labels(self):
        self.labels = self.clustering.labels_
        self.ui.textBrowser_infoPanel.append("\n##### CLUSTERING LABELS #####")
        self.ui.textBrowser_infoPanel.append(str(self.labels))
        print("\n##### CLUSTERING LABELS #####")
        print(self.labels)

    def plot_clustering(self):
        plt.clf()
        for index, i in enumerate(self.cluster.clusters):
            plt.scatter(i.center_point[0], i.center_point[1], color="red", marker="x", s=130)

        for i, (X, Y) in enumerate(self.data):
            plt.scatter(X, Y, color=self.colorspace[self.labels[i] % len(self.colorspace)])
            plt.annotate(i, (X, Y))

        for cluster in self.cluster.clusters:
            plt.scatter(cluster.central_node[0], cluster.central_node[1], color='red')

        plt.savefig("resources/temp/output.png")
        # plt.show()


class ClusterKMeans(Clustering):
    def __init__(self, data, params, ui):
        super().__init__(data, params, ui)

    def print_params(self):
        self.ui.textBrowser_infoPanel.append("\n##### K-MEANS PARAMETERS #####")
        self.ui.textBrowser_infoPanel.append(str(self.params))
        print("\n##### K-MEANS PARAMETERS #####")
        print(self.params)


    def get_clustering(self):
        self.clustering = KMeans(n_clusters=self.params.n_clusters, init=self.params.init, max_iter=self.params.max_iter, algorithm=self.params.algorithm).fit(self.data)
        self.ui.textBrowser_infoPanel.append("\n##### K-MEANS #####")
        self.ui.textBrowser_infoPanel.append(str(self.clustering))
        print("\n##### KMEANS #####")
        print(self.clustering)


class ClusterAffinity(Clustering):
    def __init__(self, data, params, ui):
        super().__init__(data, params, ui)

    def print_params(self):
        self.ui.textBrowser_infoPanel.append("\n##### AFFINITY PROPAGATION PARAMETERS #####")
        self.ui.textBrowser_infoPanel.append(str(self.params))
        print("\n##### AFFINITY PROPAGATION PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = AffinityPropagation(damping=self.params.damping, max_iter=self.params.max_iter, convergence_iter=self.params.convergence_iter,
                             affinity=self.params.affinity, random_state=self.params.random_state).fit(self.data)
        self.ui.textBrowser_infoPanel.append("\n##### AFFINITY PROPAGATION #####")
        self.ui.textBrowser_infoPanel.append(str(self.clustering))
        print("\n##### AFFINITY PROPAGATION #####")
        print(self.clustering)


class ClusterMeanShift(Clustering):
    def __init__(self, data, params, ui):
        super().__init__(data, params, ui)

    def print_params(self):
        self.ui.textBrowser_infoPanel.append("\n##### MEAN SHIFT PARAMETERS #####")
        self.ui.textBrowser_infoPanel.append(str(self.params))
        print("\n##### MEAN SHIFT PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = MeanShift(bandwidth=self.params.bandwidth, max_iter=self.params.max_iter, cluster_all=self.params.cluster_all).fit(self.data)
        self.ui.textBrowser_infoPanel.append("\n##### MEAN SHIFT #####")
        self.ui.textBrowser_infoPanel.append(str(self.clustering))
        print("\n##### MEAN SHIFT #####")
        print(self.clustering)


class ClusterSpectral(Clustering):
    def __init__(self, data, params, ui):
        super().__init__(data, params, ui)

    def print_params(self):
        self.ui.textBrowser_infoPanel.append("\n##### SPECTRAL CLUSTERING PARAMETERS #####")
        self.ui.textBrowser_infoPanel.append(str(self.params))
        print("\n##### SPECTRAL CLUSTERING PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = SpectralClustering(n_clusters=self.params.n_clusters, n_components=self.params.n_components, n_init=self.params.n_init, assign_labels=self.params.assign_labels).fit(self.data)
        self.ui.textBrowser_infoPanel.append("\n##### SPECTRAL CLUSTERING #####")
        self.ui.textBrowser_infoPanel.append(str(self.clustering))
        print("\n##### SPECTRAL CLUSTERING #####")
        print(self.clustering)


class ClusterHierarchical(Clustering):
    def __init__(self, data, params, ui):
        super().__init__(data, params, ui)

    def print_params(self):
        self.ui.textBrowser_infoPanel.append("\n##### HIERARCHICAL CLUSTERING PARAMETERS #####")
        self.ui.textBrowser_infoPanel.append(str(self.params))
        print("\n##### HIERARCHICAL CLUSTERING PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = AgglomerativeClustering(n_clusters=self.params.n_clusters, affinity=self.params.affinity, linkage=self.params.linkage).fit(self.data)
        self.ui.textBrowser_infoPanel.append("\n##### HIERARCHICAL CLUSTERING #####")
        self.ui.textBrowser_infoPanel.append(str(self.clustering))
        print("\n##### HIERARCHICAL CLUSTERING #####")
        print(self.clustering)

class ClusterDBSCAN(Clustering):
    def __init__(self, data, params, ui):
        super().__init__(data, params, ui)

    def print_params(self):
        self.ui.textBrowser_infoPanel.append("\n##### DBSCAN CLUSTERING PARAMETERS #####")
        self.ui.textBrowser_infoPanel.append(str(self.params))
        print("\n##### DBSCAN PARAMETERS #####")
        print(self.params)

    def get_clustering(self):
        self.clustering = DBSCAN(eps=self.params.eps, min_samples=self.params.min_samples, algorithm=self.params.algorithm, p=self.params.p).fit(self.data)
        self.ui.textBrowser_infoPanel.append("\n##### DBSCAN CLUSTERING #####")
        self.ui.textBrowser_infoPanel.append(str(self.clustering))
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