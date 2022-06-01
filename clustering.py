# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:01:26 2022

@author: emreg
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

class Clustering:
    
    def __init__(self):
        self.colorspace = np.array(["orange","purple","beige","brown","gray","cyan","magenta","red","green","blue","yellow","pink"])
        self.plot = plt
        
    def get_params(self): # override this function
        ...
        

class cluster_KMeans(Clustering):
    """get_data --> find_labels --> plot_clusters --> get_central_points --> plot_central_points"""
    def __init__(self):
        super().__init__()
        self.data = None
        self.labels = None
        self.central_points = None
        self.kmeans = None
        self.central_nodes = list()
        self.clusters = list()
        self.n_clusters = 3
        self.get_data()
        self.get_kmeans()
        self.find_labels()
        self.get_central_points()
        self.plot_center_points()
        self.plot_clusters()
        
        self.split_clusters()
        self.find_central_nodes()
        self.plot_central_nodes()
        
        self.__post_init__()

    def get_data(self): # --> np.array
        self.data = np.loadtxt("10.txt")
        print("DATA :")
        print(self.data)
    
    def get_params(self, init="k-means++", max_iter=300, algorithm="auto"):
        self.n_clusters = 3
        init = "k-means++"
        max_iter = 300
        algorithm = "auto"
        return self.n_clusters, init, max_iter, algorithm
    
    def get_central_points(self):
        self.central_points = self.kmeans.cluster_centers_
    
    def plot_center_points(self):
        print(self.labels)
        for index, i in enumerate(self.central_points):
            self.plot.scatter(i[0],i[1],color="red",marker="x",s=130)
            self.plot.text(i[0],i[1],s=str(index))
        
    def plot_clusters(self):
        self.plot.scatter(self.data[:,0], self.data[:,1], c=self.colorspace[0:len(self.data)])
        for i in range(0,len(self.data)):
            self.plot.annotate(i,(self.data[i,0],self.data[i,1]))
    
    def plot_central_nodes(self):
        self.plot.scatter(self.data[:,0], self.data[:,1], c=self.colorspace[0:len(self.data)])
        for i in range(self.n_clusters):
            self.plot.scatter(self.clusters[i][self.central_nodes[i]][0], self.clusters[i][self.central_nodes[i]][1], c="black")
    
    def get_kmeans(self):
        n_clusters, init, max_iter, algorithm = self.get_params()
        self.kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, algorithm=algorithm).fit(self.data)
    
    def find_labels(self):
        self.labels = self.kmeans.labels_
        print("LABELS:")
        print(self.labels)
        print("KMEANS :")
        print(self.kmeans)
    
    def split_clusters(self):
        for i in range(self.n_clusters):
            print(self.data[self.labels == i])
            self.clusters.append(self.data[self.labels == i])
        print("SPLIT")
        print(self.clusters)
    
    def print_clusters(self):
        unique, indice = np.unique(self.labels, return_index=True)
        index=list()
        liste=[]
        print(f"There are {len(unique)} clusters")
        for i in unique:
            index=list((np.where(self.labels == unique[i]))[0])
            liste.append(index)
            print(f"Cluster {unique[i]} ======> {index}")
    
    def find_central_nodes(self):
        for i in range(self.n_clusters):
            diff = (self.central_points[i] - self.clusters[i])
            dist = np.sqrt(np.sum(diff**2,axis=-1))
            self.central_nodes.append(np.argmin(dist))
        print("CLuster Central Nodes : ")
        print(self.central_nodes)
        
    
    def __post_init__(self):
        self.plot.show()
        self.print_clusters()


if __name__ == '__main__':
    kmeans = cluster_KMeans()
        