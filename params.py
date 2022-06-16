from dataclasses import dataclass

## Dataclass for holding parameters of clustering algorithm
#  Initialized by default values
# @param n_clusters The number of clusters to form as well as the number of centroids to generate.
# @param init Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# @param max_iter Maximum number of iterations of the k-means algorithm for a single run.
# @param algorithm K-means algorithm to use.
@dataclass
class KMeansParams:
    n_clusters: int = 3
    init: str = "k-means++"
    max_iter: int = 300
    algorithm: str = "auto"


## Dataclass for holding parameters of clustering algorithm
#  Initialized by default values
# @param damping Damping factor in the range [0.5, 1.0) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping).
# @param max_iter Maximum number of iterations.
# @param convergence_iter Number of iterations with no change in the number of estimated clusters that stops the convergence.
# @param affinity Which affinity to use.
# @param random_state Pseudo-random number generator to control the starting state.
@dataclass
class AffinityParams:
    damping: float = 0.5
    max_iter: int = 200
    convergence_iter: int = 15
    affinity: str = 'euclidean'
    random_state: int = None


## Dataclass for holding parameters of clustering algorithm
#  Initialized by default values
# @param bandwidth Bandwidth used in the RBF kernel.
# @param max_iter Maximum number of iterations, per seed point before the clustering operation terminates (for that seed point), if has not converged yet.
# @param cluster_all If true, then all points are clustered, even those orphans that are not within any kernel.
@dataclass
class MeanShiftParams:
    bandwidth: float = None
    max_iter: int = 300
    cluster_all: bool = True


## Dataclass for holding parameters of clustering algorithm
#  Initialized by default values
# @param n_clusters The dimension of the projection subspace.
# @param n_components Number of eigenvectors to use for the spectral embedding.
# @param n_init Number of time the k-means algorithm will be run with different centroid seeds.
# @param assign_labels The strategy for assigning labels in the embedding space.
@dataclass
class SpectralParams:
    n_clusters: int = 3
    n_components: int = None
    n_init: int = 10
    assign_labels: str = 'kmeans'


## Dataclass for holding parameters of clustering algorithm
#  Initialized by default values
# @param n_clusters The number of clusters to find.
# @param affinity Metric used to compute the linkage.
# @param linkage Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation.
@dataclass
class HierarchicalParams:
    n_clusters: int = 2
    affinity: str = 'euclidean'
    linkage: str = 'ward'


## Dataclass for holding parameters of clustering algorithm
#  Initialized by default values
# @param eps The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# @param min_samples The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
# @param algorithm The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.
# @param p The power of the Minkowski metric to be used to calculate distance between points.
@dataclass
class DBSCANParams:
    eps: float = 0.5
    min_samples: int = 5
    algorithm: str = 'auto'
    p: float = 2
