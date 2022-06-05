from PyQt5 import QtGui, QtWidgets
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from k_means_params import Ui_Form as Ui_Form_K_Means
from affinity_params import Ui_Form as Ui_Form_Affinity
from dbscan_params import Ui_Form as Ui_Form_DBSCAN
from hierarchical_params import Ui_Form as Ui_Form_Hierarchical
from spectral_params import Ui_Form as Ui_Form_Spectral
from mean_shift_params import Ui_Form as Ui_Form_MeanShift
from clustering_v2 import ClusterKMeans, ClusterAffinity, ClusterDBSCAN, ClusterMeanShift, ClusterSpectral, ClusterHierarchical
from params import KMeansParams, MeanShiftParams, SpectralParams, AffinityParams, DBSCANParams, HierarchicalParams
from skimage import io
from typing import Protocol
from heuristics import HillClimbing


class Driver:
    def __init__(self, ui):
        self.ui = ui
        self.input_image = None
        self.output_image = None
        self.data = None
        self.data = None
        self.save_path = None
        self.cluster_holder = None
        self.setup_icons()
        self.setup_signal_slots()
        self.create_widgets()

    def exit(self):
        QtWidgets.QApplication.quit()

    def setup_signal_slots(self):
        self.ui.actionOpen_Data.triggered.connect(self.open_data)
        self.ui.actionSave_Initial_Solution.triggered.connect(self.save_initial_solution)
        self.ui.actionSave_Final_Solution.triggered.connect(self.save_final_solution)
        self.ui.actionExit.triggered.connect(self.exit)
        self.ui.actionClear_Initial_Solution.triggered.connect(self.clear_initial_solution)
        self.ui.actionClear_Final_Solution.triggered.connect(self.clear_final_solution)
        self.ui.actionUndo_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionUndo_Final_Solution.triggered.connect(self.say_hello)
        self.ui.actionRedo_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionRedo_Final_Solution.triggered.connect(self.say_hello)
        self.ui.actionK_Means.triggered.connect(self.open_k_means)
        self.ui.actionAffinity_Propagation.triggered.connect(self.open_affinity)
        self.ui.actionMean_shift.triggered.connect(self.open_meanshift)
        self.ui.actionSpectral_Clustering.triggered.connect(self.open_spectral)
        self.ui.actionHierarchical_Clustering.triggered.connect(self.open_hierarchical)
        self.ui.actionDBSCAN.triggered.connect(self.open_dbscan)
        self.ui.actionHill_Climbing.triggered.connect(self.hill_climbing)
        self.ui.actionSimulated_Anneling.triggered.connect(self.say_hello)
        self.ui.actionExport_As_Initial_Solution.triggered.connect(self.export_as_initial_solution)
        self.ui.actionExport_As_Final_Solution.triggered.connect(self.export_as_final_solution)

    def hill_climbing(self):
        HillClimbing(self.cluster_holder, n_iterations=1000)
        self.plot_final_solution()

    def save_initial_solution(self):
        ...

    def save_final_solution(self):
        ...

    def clear_initial_solution(self):
        if self.input_image is not None:
            self.ui.label_initialSolution.clear()
            self.input_image = None

    def clear_final_solution(self):
        if self.output_image is not None:
            self.ui.label_finalSolution.clear()
            self.output_image = None

    def export_as_initial_solution(self):
        if self.input_image is not None:
            self.save_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Image files (*.jpg)")
            if len(self.save_path) != 0:
                output = self.input_image[:, :, :-1] # RGBA to RGB before saving (JPG DOES NOT SUPPORT RGBA)
                io.imsave(self.save_path, output)
                # self.check_save_buttons()
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before saving.')

    def export_as_final_solution(self):
        if self.output_image is not None:
            self.save_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Image files (*.jpg)")
            if len(self.save_path) != 0:
                output = self.output_image[:, :, :-1] # RGBA to RGB before saving (JPG DOES NOT SUPPORT RGBA)
                io.imsave(self.save_path, output)
                # self.check_save_buttons()
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before saving.')


    def open_data(self):
        try:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(filter="Text files (*.txt)")
            self.data = np.loadtxt(fname)
            print(self.data)
            self.plot_initial_solution()
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Couldn't open file.")

    def plot_initial_solution(self):
        plt.clf()
        for i, (X,Y) in enumerate(self.data):
            plt.scatter(X, Y, color='black')
            plt.annotate(i, (X, Y))
        plt.savefig("resources/temp/input.png")
        self.input_image = io.imread("resources/temp/input.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/input.png"))

    def plot_final_solution(self):
        self.output_image = io.imread("resources/temp/output.png")
        self.ui.label_finalSolution.setPixmap(QtGui.QPixmap("resources/temp/output.png"))

    def say_hello(self):
        print("HELLO THERE !")

    def open_k_means(self):
        self.kmeans_widget.show()

    def k_means(self):
        n_clusters = int(self.kmeans_ui.n_clusters.text())
        init = self.kmeans_ui.init.currentText()
        max_iter = int(self.kmeans_ui.max_iter.text())
        algorithm = self.kmeans_ui.algorithm.currentText()
        operation = ClusterKMeans(self.data, KMeansParams(n_clusters, init, max_iter, algorithm), self.ui)
        self.cluster_holder = operation.get_cluster_holder()
        self.kmeans_widget.hide()
        self.output_image = io.imread("resources/temp/input.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/input.png"))

    def open_affinity(self):
        self.affinity_widget.show()

    def affinity(self):
        damping = float(self.affinity_ui.damping.text())
        max_iter = int(self.affinity_ui.max_iter.text())
        convergence_iter = int(self.affinity_ui.convergence_iter.text())
        affinity = self.affinity_ui.affinity.currentText()
        random_state = int(self.affinity_ui.random_state.text())
        operation = ClusterAffinity(self.data, AffinityParams(damping, max_iter, convergence_iter, affinity, random_state), self.ui)
        self.cluster_holder = operation.get_cluster_holder()
        self.affinity_widget.hide()
        self.output_image = io.imread("resources/temp/input.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/input.png"))

    def open_dbscan(self):
        self.dbscan_widget.show()

    def dbscan(self):
        eps = float(self.dbscan_ui.eps.text())
        min_samples = int(self.dbscan_ui.min_samples.text())
        algorithm = self.dbscan_ui.algorithm.currentText()
        p = float(self.dbscan_ui.p.text())
        operation = ClusterDBSCAN(self.data, DBSCANParams(eps, min_samples, algorithm, p), self.ui)
        self.cluster_holder = operation.get_cluster_holder()
        self.dbscan_widget.hide()
        self.output_image = io.imread("resources/temp/input.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/input.png"))

    def open_hierarchical(self):
        self.hierarchical_widget.show()

    def hierarchical(self):
        n_clusters = int(self.hierarchical_ui.n_clusters.text())
        affinity = self.hierarchical_ui.affinity.currentText()
        linkage = self.hierarchical_ui.linkage.currentText()
        operation = ClusterHierarchical(self.data, HierarchicalParams(n_clusters, affinity, linkage), self.ui)
        self.cluster_holder = operation.get_cluster_holder()
        self.hierarchical_widget.hide()
        self.output_image = io.imread("resources/temp/input.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/input.png"))

    def open_meanshift(self):
        self.meanshift_widget.show()

    def meanshift(self):
        bandwidth = float(self.meanshift_ui.bandwidth.text())
        max_iter = int(self.meanshift_ui.bandwidth.text())
        cluster_all = self.meanshift_ui.cluster_all.currentText()
        if cluster_all == 'True':
            cluster_all = True
        else:
            cluster_all = False
        operation = ClusterMeanShift(self.data, MeanShiftParams(bandwidth, max_iter, cluster_all), self.ui)
        self.cluster_holder = operation.get_cluster_holder()
        self.meanshift_widget.hide()
        self.output_image = io.imread("resources/temp/input.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/input.png"))

    def open_spectral(self):
        self.spectral_widget.show()

    def spectral(self):
        n_clusters = int(self.spectral_ui.n_clusters.text())
        n_components = int(self.spectral_ui.n_components.text())
        n_init = int(self.spectral_ui.n_init.text())
        assign_labels = self.spectral_ui.assign_labels.currentText()
        operation = ClusterSpectral(self.data, SpectralParams(n_clusters, n_components, n_init, assign_labels), self.ui)
        self.cluster_holder = operation.get_cluster_holder()
        self.spectral_widget.hide()
        self.output_image = io.imread("resources/temp/input.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/input.png"))

    def create_widgets(self):
        self.kmeans_widget = QtWidgets.QWidget()
        self.kmeans_ui = Ui_Form_K_Means()
        self.kmeans_ui.setupUi(self.kmeans_widget)
        self.kmeans_ui.OKButton.clicked['bool'].connect(self.k_means)

        self.affinity_widget = QtWidgets.QWidget()
        self.affinity_ui = Ui_Form_Affinity()
        self.affinity_ui.setupUi(self.affinity_widget)
        self.affinity_ui.OKButton.clicked['bool'].connect(self.affinity)

        self.dbscan_widget = QtWidgets.QWidget()
        self.dbscan_ui = Ui_Form_DBSCAN()
        self.dbscan_ui.setupUi(self.dbscan_widget)
        self.dbscan_ui.OKButton.clicked['bool'].connect(self.dbscan)

        self.hierarchical_widget = QtWidgets.QWidget()
        self.hierarchical_ui = Ui_Form_Hierarchical()
        self.hierarchical_ui.setupUi(self.hierarchical_widget)
        self.hierarchical_ui.OKButton.clicked['bool'].connect(self.hierarchical)

        self.meanshift_widget = QtWidgets.QWidget()
        self.meanshift_ui = Ui_Form_MeanShift()
        self.meanshift_ui.setupUi(self.meanshift_widget)
        self.meanshift_ui.OKButton.clicked['bool'].connect(self.meanshift)

        self.spectral_widget = QtWidgets.QWidget()
        self.spectral_ui = Ui_Form_Spectral()
        self.spectral_ui.setupUi(self.spectral_widget)
        self.spectral_ui.OKButton.clicked['bool'].connect(self.spectral)

    def setup_icons(self):
        self.ui.toolButton_openData.setIcon(QtGui.QIcon("resources/icons/open.png"))
        self.ui.toolButton_saveInitialSolution.setIcon(QtGui.QIcon("resources/icons/save.png"))
        self.ui.toolButton_exportAsInitialSolution.setIcon(QtGui.QIcon("resources/icons/saveas.png"))
        self.ui.toolButton_clearInitialSolution.setIcon(QtGui.QIcon("resources/icons/swap.png"))
        self.ui.toolButton_undoInitialSolution.setIcon(QtGui.QIcon("resources/icons/clear_input.png"))
        self.ui.toolButton_redoInitialSolution.setIcon(QtGui.QIcon("resources/icons/clear_output.png"))
        self.ui.toolButton_saveFinalSolution.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.toolButton_exportAsFinalSolution.setIcon(QtGui.QIcon("resources/icons/redo.png"))
        self.ui.toolButton_clearFinalSolution.setIcon(QtGui.QIcon("resources/icons/grayscale.png"))
        self.ui.toolButton_undoFinalSolution.setIcon(QtGui.QIcon("resources/icons/hsv.png"))
        self.ui.toolButton_redoFinalSolution.setIcon(QtGui.QIcon("resources/icons/multiotsu.png"))
        self.ui.toolButton_kMeans.setIcon(QtGui.QIcon("resources/icons/chanvese.png"))
        self.ui.toolButton_affinityPropagation.setIcon(QtGui.QIcon("resources/icons/acwe.png"))
        self.ui.toolButton_meanShift.setIcon(QtGui.QIcon("resources/icons/gac.png"))
        self.ui.toolButton_spectralClustering.setIcon(QtGui.QIcon("resources/icons/roberts.png"))
        self.ui.toolButton_hierarchicalClustering.setIcon(QtGui.QIcon("resources/icons/sobel.png"))
        self.ui.toolButton_dbscan.setIcon(QtGui.QIcon("resources/icons/scharr.png"))
        self.ui.toolButton_hillClimbing.setIcon(QtGui.QIcon("resources/icons/prewitt.png"))
        self.ui.toolButton_simulatedAnneling.setIcon(QtGui.QIcon("resources/icons/exit.png"))
        self.ui.toolButton_exit.setIcon(QtGui.QIcon("resources/icons/exit.png"))

        self.ui.actionOpen_Data.setIcon(QtGui.QIcon("resources/icons/open.png"))
        self.ui.actionSave_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/save.png"))
        self.ui.actionSave_Final_Solution.setIcon(QtGui.QIcon("resources/icons/saveas.png"))
        self.ui.actionExit.setIcon(QtGui.QIcon("resources/icons/exit.png"))
        self.ui.actionClear_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/swap.png"))
        self.ui.actionClear_Final_Solution.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.actionUndo_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/redo.png"))
        self.ui.actionUndo_Final_Solution.setIcon(QtGui.QIcon("resources/icons/input.png"))
        self.ui.actionRedo_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/output.png"))
        self.ui.actionRedo_Final_Solution.setIcon(QtGui.QIcon("resources/icons/grayscale.png"))
        self.ui.actionK_Means.setIcon(QtGui.QIcon("resources/icons/hsv.png"))
        self.ui.actionAffinity_Propagation.setIcon(QtGui.QIcon("resources/icons/multiotsu.png"))
        self.ui.actionMean_shift.setIcon(QtGui.QIcon("resources/icons/chanvese.png"))
        self.ui.actionSpectral_Clustering.setIcon(QtGui.QIcon("resources/icons/roberts.png"))
        self.ui.actionHierarchical_Clustering.setIcon(QtGui.QIcon("resources/icons/sobel.png"))
        self.ui.actionDBSCAN.setIcon(QtGui.QIcon("resources/icons/scharr.png"))
        self.ui.actionHill_Climbing.setIcon(QtGui.QIcon("resources/icons/prewitt.png"))
        self.ui.actionSimulated_Anneling.setIcon(QtGui.QIcon("resources/icons/acwe.png"))
        self.ui.actionExport_As_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/gac.png"))
        self.ui.actionExport_As_Final_Solution.setIcon(QtGui.QIcon("resources/icons/gac.png"))

        self.ui.menuClear.setIcon(QtGui.QIcon("resources/icons/clear.png"))
