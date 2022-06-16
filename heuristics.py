from matplotlib import pyplot as plt
from numpy.random import rand, randn, randint
from typing import Protocol
from utils import ClusterHolder
from copy import deepcopy
import numpy as np

## Base class for Heuristics
class Heuristics:
	def __init__(self, ch: ClusterHolder, n_iterations: int):
		self.info = ""
		self.colorspace = ["purple", "cyan", "green", "orange", "brown", "gray", "magenta", "blue", "yellow", "pink"]
		self.ch_origin = deepcopy(ch)
		self.ch = ch
		self.n_clusters = self.ch.n_clusters  # number of clusters
		self.n_iterations = n_iterations
		self.initial_score = max(self.ch.objective_function)
		self.evaluate()
		self.plot_clustering()

	## When operation is done, by calling this function we get new clusters
	#  @return Manipulated (optimized) clusters. --> ClusterHolder object.
	def get_final_solution(self):
		self.ch_origin.rewrite_info()
		return self.ch_origin

	## Override this function
	def evaluate(self): # override this function
		...

	## Change the location of the hub in a randomly chosen cluster with a randomly chosen node in the same cluster
	def relocate_hub(self):
		rand_cl = self.rand_cluster()
		rand_cl.central_node_index = randint(len(rand_cl.points))
		rand_cl.central_node = rand_cl.points[rand_cl.central_node_index]

	## From a randomly chosen cluster, change the allocation of non-hub node to a different randomly chosen cluster.
	#  If the randomly chosen cluster consists of only one node, we do not allow this operation.
	def reallocate_node(self):  
		rand_cl = self.rand_cluster()
		rand_cl2 = self.rand_cluster()
		r_node_index, r_node = self.rand_node(rand_cl)
		if rand_cl != rand_cl2:
			if r_node_index != rand_cl.central_node_index:
				rand_cl.points = np.delete(rand_cl.points, r_node_index,0)
				if rand_cl.central_node_index > r_node_index:
					rand_cl.central_node_index -= 1
				rand_cl2.points = np.append(rand_cl2.points, [r_node], axis=0)
		rand_cl.central_node = rand_cl.points[rand_cl.central_node_index]


	## Swap the allocations of two randomly chosen non-hub nodes from different clusters.
	def swap_nodes(self):
		rand_cl = self.rand_cluster()
		r_node_index, r_node = self.rand_node(rand_cl)
		rand_cl2 = self.rand_cluster()
		r_node2_index, r_node2 = self.rand_node(rand_cl2)
		if rand_cl != rand_cl2:
			if r_node_index != rand_cl.central_node_index and r_node2_index != rand_cl2.central_node_index:
				rand_cl.points = np.delete(rand_cl.points, r_node_index, 0)
				if rand_cl.central_node_index > r_node_index:
					rand_cl.central_node_index -= 1
				rand_cl2.points = np.delete(rand_cl2.points, r_node2_index, 0)
				if rand_cl2.central_node_index > r_node2_index:
					rand_cl2.central_node_index -= 1
				rand_cl.points = np.append(rand_cl.points, [r_node2], axis=0)
				rand_cl2.points = np.append(rand_cl2.points, [r_node], axis=0)
		rand_cl.central_node = rand_cl.points[rand_cl.central_node_index]
		rand_cl2.central_node = rand_cl2.points[rand_cl2.central_node_index]

	## Pick a random cluster and return it
	def rand_cluster(self):
		return self.ch.clusters[randint(self.n_clusters)]

	## Pick a random node in given cluster and return it and its index
	def rand_node(self, cluster):
		index = randint(len(cluster.points))
		return index, cluster.points[index]

	## Plot final solution that improved from initial solution
	def plot_clustering(self):
		plt.clf()
		for index, cluster in enumerate(self.ch_origin.clusters):
			plt.scatter(cluster.center_point[0], cluster.center_point[1], color="red", marker="x", s=130)
			for i, (X,Y) in enumerate(cluster.points):
				plt.scatter(X, Y, color=self.colorspace[cluster.cluster_index % len(self.colorspace)])
				plt.annotate(self.ch.get_data_index_by_point([X,Y]), (X,Y))
			plt.scatter(cluster.central_node[0], cluster.central_node[1], color='red')

		plt.savefig("resources/temp/output.png")


## HillClimbing optimization algorithm
class HillClimbing(Heuristics):
	## Constructor
	# @param n_iterations Number of iterations.
	def __init__(self, ch, n_iterations):
		super().__init__(ch, n_iterations)

	## Run --> hill climbing
	def evaluate(self):
		self.info += f"\nRunning hill-climbing algorithm for {self.n_iterations} times\n"
		solution_eval = self.initial_score
		self.info += f"Initial score is {self.initial_score}\n\n"
		for i in range(self.n_iterations):
			# take a step
			self.relocate_hub()  # --> do some modifications and find new solution candidate
			self.reallocate_node()
			self.swap_nodes()
			# evaluate candidate point
			candidate_eval = self.ch.calculate_objective_function()  # --> get objective function score from it
			# check if we should keep the new point
			if candidate_eval < solution_eval:  # --> if candidate_score is lower than initial score (we will use greater than)
				# store the new point
				self.ch_origin = deepcopy(self.ch)
				solution_eval = candidate_eval
				# report progress
				self.info += f"Iteration({i}) - New solution found, new score --> {solution_eval:.3f}\n"  # --> report the progress
			else:
				self.ch = deepcopy(self.ch_origin)
		self.info += f"\nInitial score --> {self.initial_score}, New score --> {solution_eval}\n"
		# return [solution, solution_eval]


## SimulatedAnnealing optimization algorithm
class SimulatedAnnealing(Heuristics):
	## Constructor
	# @param n_iterations Number of iterations.
	def __init__(self, ch, n_iterations):
		super().__init__(ch, n_iterations)

	## Run --> simulated annealing
	def evaluate(self):
		...
