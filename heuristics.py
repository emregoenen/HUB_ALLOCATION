from matplotlib import pyplot as plt
from numpy.random import rand, randn, randint
from typing import Protocol
from utils import ClusterHolder
from copy import deepcopy
import numpy as np


class Heuristics:
	def __init__(self, ch: ClusterHolder, n_iterations: int):
		self.colorspace = ["purple", "cyan", "green", "orange", "brown", "gray", "magenta", "blue", "yellow", "pink"]
		self.ch_origin = deepcopy(ch)
		self.ch = ch
		self.n_clusters = self.ch.n_clusters  # number of clusters
		self.n_iterations = n_iterations
		self.initial_score = max(self.ch.objective_function)
		self.evaluate()
		self.plot_clustering()

	def evaluate(self): # override this function
		...

	def relocate_hub(self):  # Change the location of the hub in a randomly chosen cluster with a randomly chosen node in the same cluster
		rand_cl = self.rand_cluster()
		rand_cl.central_node_index = randint(len(rand_cl.points))
		rand_cl.central_node = rand_cl.points[rand_cl.central_node_index]

	def reallocate_node(self):  # From a randomly chosen cluster, change the allocation of non-hub node to a different randomly chosen cluster.
		# If the randomly chosen cluster consists of only one node, we do not allow this operation.
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


	def swap_nodes(self):  # Swap the allocations of two randomly chosen non-hub nodes from different clusters.
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

	def rand_cluster(self):
		return self.ch.clusters[randint(self.n_clusters)]

	def rand_node(self, cluster):
		index = randint(len(cluster.points))
		return index, cluster.points[index]

	def plot_clustering(self):
		plt.clf()
		for index, cluster in enumerate(self.ch.clusters):
			plt.scatter(cluster.center_point[0], cluster.center_point[1], color="red", marker="x", s=130)
			for i, (X,Y) in enumerate(cluster.points):
				plt.scatter(X, Y, color=self.colorspace[cluster.cluster_index % len(self.colorspace)])
				plt.annotate(self.ch.get_data_index_by_point([X,Y]), (X,Y))
			plt.scatter(cluster.central_node[0], cluster.central_node[1], color='red')

		plt.savefig("resources/temp/output.png")


class HillClimbing(Heuristics):
	def __init__(self, ch, n_iterations):
		super().__init__(ch, n_iterations)

	def evaluate(self):
		solution_eval = self.initial_score
		for i in range(self.n_iterations):
			# take a step
			self.relocate_hub()  # --> do some modifications and find new solution candidate
			self.reallocate_node()
			self.swap_nodes()
			# evaluate candidate point
			candidate_eval = self.ch.calculate_objective_function()  # --> get objective function score from it
			# check if we should keep the new point
			if candidate_eval > solution_eval:  # --> if candidate_score is lower than initial score (we will use greater than)
				# store the new point
				self.ch_origin = deepcopy(self.ch)
				solution_eval = candidate_eval
				# report progress
				print('>%d f(%s) = %.5f\n\n' % (i, self.ch, solution_eval))  # --> report he progress
			else:
				self.ch = deepcopy(self.ch_origin)
		print("Initial was --> ", self.initial_score, "new score --> ", solution_eval)
		# return [solution, solution_eval]

class SimulatedAnneling(Heuristics):
	def __init__(self, ch, n_iterations):
		super().__init__(ch, n_iterations)


# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):

	# generate an initial point														--> get initial clusters
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point													--> do not evaluate (we have initial solution score)
	solution_eval = objective(solution)
	# run the hill climb
	for i in range(n_iterations):
		# take a step
		candidate = solution + randn(len(bounds)) * step_size					#	--> do some modifications and find new solution candidate
		# evaluate candidate point
		candidte_eval = objective(candidate)									#	--> get objective function score from it
		# check if we should keep the new point
		if candidte_eval <= solution_eval:										#	--> if candidate_score is lower than initial score (we will use greater than)
			# store the new point
			solution, solution_eval = candidate, candidte_eval					#	--> store the clusters and cluster score
			# report progress
			print('>%d f(%s) = %.5f' % (i, solution, solution_eval))			#	--> report he progress
	return [solution, solution_eval]