from dataclasses import dataclass

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


@dataclass
class MeanShiftParams:
    bandwidth: float = None
    max_iter: int = 300
    cluster_all: bool = True


@dataclass
class SpectralParams:
    n_clusters: int = 3
    n_components: int = None
    n_init: int = 10
    assign_labels: str = 'kmeans'

@dataclass
class HierarchicalParams:
    n_clusters: int = 2
    affinity: str = 'euclidean'
    linkage: str = 'ward'

@dataclass
class DBSCANParams:
    eps: float = 0.5
    min_samples: int = 5
    algorithm: str = 'auto'
    p: float = 2
