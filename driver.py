from PyQt5 import QtGui, QtWidgets
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from paramwidgets.k_means_params import Ui_Form as Ui_Form_K_Means
from paramwidgets.affinity_params import Ui_Form as Ui_Form_Affinity
from paramwidgets.dbscan_params import Ui_Form as Ui_Form_DBSCAN
from paramwidgets.hierarchical_params import Ui_Form as Ui_Form_Hierarchical
from paramwidgets.spectral_params import Ui_Form as Ui_Form_Spectral
from paramwidgets.mean_shift_params import Ui_Form as Ui_Form_MeanShift
from clustering import ClusterKMeans, ClusterAffinity, ClusterDBSCAN, ClusterMeanShift, ClusterSpectral, ClusterHierarchical
from params import KMeansParams, MeanShiftParams, SpectralParams, AffinityParams, DBSCANParams, HierarchicalParams
from skimage import io
from typing import Protocol
from heuristics import HillClimbing


class Driver:
    def __init__(self, ui, MainWindow):
        self.ui = ui
        self.MainWindow = MainWindow
        self.input_image = None
        self.output_image = None
        self.data = None
        self.data = None
        self.save_path = None
        self.cluster_holder = None
        self.info = ""
        self.setup_icons()
        self.setup_signal_slots()
        self.create_widgets()

    def exit(self):
        if self.ui.toolButton_exportAsFinalSolution.isEnabled():
            reply = QtWidgets.QMessageBox.question(self.MainWindow, 'Unsaved Changes',
                                                   "You have unsaved changes. Do you want to save them ?",
                                                   QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.export_as_final_solution()

        reply = QtWidgets.QMessageBox.question(self.MainWindow, 'Quit',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
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
        self.check_buttons()
        # self.check_save_export_as_buttons()
        # self.check_undo_redo_buttons()
        # self.check_clear_buttons()

    def save_initial_solution(self):
        ...

    def save_final_solution(self):
        ...

    def clear_initial_solution(self):
        if self.input_image is not None:
            self.ui.label_initialSolution.clear()
            self.input_image = None
        self.clear_final_solution()
        self.check_buttons()

    def clear_final_solution(self):
        if self.output_image is not None:
            self.ui.label_finalSolution.clear()
            self.output_image = None
        self.check_buttons()

    def export_as_initial_solution(self):
        if self.input_image is not None:
            self.save_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Image files (*.jpg)")
            if len(self.save_path) != 0:
                output = self.input_image[:, :, :-1] # RGBA to RGB before saving (JPG DOES NOT SUPPORT RGBA)
                io.imsave(self.save_path, output)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before saving.')

    def export_as_final_solution(self):
        if self.output_image is not None:
            self.save_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Image files (*.jpg)")
            if len(self.save_path) != 0:
                output = self.output_image[:, :, :-1] # RGBA to RGB before saving (JPG DOES NOT SUPPORT RGBA)
                io.imsave(self.save_path, output)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before saving.')

    def open_data(self):
        try:
            self.print_info("Opening data...")
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(filter="Text files (*.txt)")
            self.data = np.loadtxt(fname)

            self.print_info("### DATA ###")
            self.print_info(self.data)

            self.plot_initial_solution()
            self.check_buttons()
            # self.check_clustering_buttons()
            # self.check_clear_buttons()
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Couldn't open file.")

    def print_info(self, info):
        self.ui.textBrowser_infoPanel.append(str(info))

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
        try:
            data = self.kmeans_ui.n_clusters.text()
            n_clusters = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for n_clusters. --> integer")
            return -1

        init = self.kmeans_ui.init.currentText()

        try:
            data = self.kmeans_ui.max_iter.text()
            max_iter = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for max_iter. --> integer")
            return -1

        algorithm = self.kmeans_ui.algorithm.currentText()
        operation = ClusterKMeans(self.data, KMeansParams(n_clusters, init, max_iter, algorithm), self.ui, self)
        self.cluster_holder = operation.get_cluster_holder()
        self.kmeans_widget.hide()
        self.print_info(self.cluster_holder.info)

    def open_affinity(self):
        self.affinity_widget.show()

    def affinity(self):
        try:
            data = self.affinity_ui.damping.text()
            damping = float(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for damping. --> float")
            return -1

        try:
            data = self.affinity_ui.max_iter.text()
            max_iter = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for max_iter. --> integer")
            return -1

        try:
            data = self.affinity_ui.convergence_iter.text()
            convergence_iter = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for convergence_iter. --> integer")
            return -1

        affinity = self.affinity_ui.affinity.currentText()
        try:
            data = self.affinity_ui.random_state.text()
            if data == 'None':
                random_state = None
            else:
                random_state = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for random_state. --> integer or None")
            return -1

        operation = ClusterAffinity(self.data, AffinityParams(damping, max_iter, convergence_iter, affinity, random_state), self.ui, self)
        self.cluster_holder = operation.get_cluster_holder()
        self.affinity_widget.hide()
        self.print_info(self.cluster_holder.info)

    def open_dbscan(self):
        self.dbscan_widget.show()

    def dbscan(self):
        try:
            data = self.dbscan_ui.eps.text()
            eps = float(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for eps. --> float")
            return -1

        try:
            data = self.dbscan_ui.min_samples.text()
            min_samples = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for min_samples. --> integer")
            return -1

        algorithm = self.dbscan_ui.algorithm.currentText()

        try:
            data = self.dbscan_ui.p.text()
            p = float(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for p. --> float")
            return -1


        operation = ClusterDBSCAN(self.data, DBSCANParams(eps, min_samples, algorithm, p), self.ui, self)
        self.cluster_holder = operation.get_cluster_holder()
        self.dbscan_widget.hide()
        self.print_info(self.cluster_holder.info)

    def open_hierarchical(self):
        self.hierarchical_widget.show()

    def hierarchical(self):
        try:
            data = self.hierarchical_ui.n_clusters.text()
            n_clusters = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for n_clusters. --> integer")
            return -1

        affinity = self.hierarchical_ui.affinity.currentText()
        linkage = self.hierarchical_ui.linkage.currentText()
        operation = ClusterHierarchical(self.data, HierarchicalParams(n_clusters, affinity, linkage), self.ui, self)
        self.cluster_holder = operation.get_cluster_holder()
        self.hierarchical_widget.hide()
        self.print_info(self.cluster_holder.info)

    def open_meanshift(self):
        self.meanshift_widget.show()

    def meanshift(self):
        try:
            data = self.meanshift_ui.bandwidth.text()
            if data == 'None':
                bandwidth = None
            else:
                bandwidth = float(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for bandwidth. --> float or None")
            return -1

        try:
            data = self.meanshift_ui.max_iter.text()
            max_iter = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for max_iter. --> integer")
            return -1

        cluster_all = self.meanshift_ui.cluster_all.currentText()
        if cluster_all == 'True':
            cluster_all = True
        else:
            cluster_all = False
        operation = ClusterMeanShift(self.data, MeanShiftParams(bandwidth, max_iter, cluster_all), self.ui, self)
        self.cluster_holder = operation.get_cluster_holder()
        self.meanshift_widget.hide()
        self.print_info(self.cluster_holder.info)

    def open_spectral(self):
        self.spectral_widget.show()

    def spectral(self):
        try:
            data = self.spectral_ui.n_clusters.text()
            n_clusters = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for n_clusters. --> integer")
            return -1

        try:
            data = self.spectral_ui.n_components.text()
            if data == 'None':
                n_components = None
            else:
                n_components = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for n_components. --> integer")
            return -1

        try:
            data = self.spectral_ui.n_init.text()
            n_init = int(data)
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Enter a valid input for n_init. --> integer")
            return -1

        assign_labels = self.spectral_ui.assign_labels.currentText()
        operation = ClusterSpectral(self.data, SpectralParams(n_clusters, n_components, n_init, assign_labels), self.ui, self)
        self.cluster_holder = operation.get_cluster_holder()
        self.spectral_widget.hide()
        self.print_info(self.cluster_holder.info)

    def check_buttons(self):
        self.check_undo_redo_buttons()
        self.check_clear_buttons()
        self.check_clustering_buttons()
        self.check_save_export_as_buttons()
        self.check_heuristics_buttons()

    def check_clear_buttons(self):
        if self.input_image is not None:
            self.ui.toolButton_clearInitialSolution.setEnabled(True)
            self.ui.actionClear_Initial_Solution.setEnabled(True)
        else:
            self.ui.toolButton_clearInitialSolution.setEnabled(False)
            self.ui.actionClear_Initial_Solution.setEnabled(False)

        if self.output_image is not None:
            self.ui.toolButton_clearFinalSolution.setEnabled(True)
            self.ui.actionClear_Final_Solution.setEnabled(True)
        else:
            self.ui.toolButton_clearFinalSolution.setEnabled(False)
            self.ui.actionClear_Final_Solution.setEnabled(False)

    def check_save_export_as_buttons(self):
        if self.input_image is not None:
            self.ui.toolButton_saveInitialSolution.setEnabled(True)
            self.ui.toolButton_exportAsInitialSolution.setEnabled(True)
            self.ui.actionSave_Initial_Solution.setEnabled(True)
            self.ui.actionExport_As_Initial_Solution.setEnabled(True)
        else:
            self.ui.toolButton_saveInitialSolution.setEnabled(False)
            self.ui.toolButton_exportAsInitialSolution.setEnabled(False)
            self.ui.actionSave_Initial_Solution.setEnabled(False)
            self.ui.actionExport_As_Initial_Solution.setEnabled(False)

        if self.output_image is not None:
            self.ui.toolButton_saveFinalSolution.setEnabled(True)
            self.ui.toolButton_exportAsFinalSolution.setEnabled(True)
            self.ui.actionSave_Final_Solution.setEnabled(True)
            self.ui.actionExport_As_Final_Solution.setEnabled(True)
        else:
            self.ui.toolButton_saveFinalSolution.setEnabled(False)
            self.ui.toolButton_exportAsFinalSolution.setEnabled(False)
            self.ui.actionSave_Final_Solution.setEnabled(False)
            self.ui.actionExport_As_Final_Solution.setEnabled(False)

    def check_heuristics_buttons(self):
        if self.input_image is not None:
            self.ui.toolButton_hillClimbing.setEnabled(True)
            self.ui.toolButton_simulatedAnneling.setEnabled(True)
            self.ui.actionHill_Climbing.setEnabled(True)
            self.ui.actionSimulated_Anneling.setEnabled(True)
        else:
            self.ui.toolButton_hillClimbing.setEnabled(False)
            self.ui.toolButton_simulatedAnneling.setEnabled(False)
            self.ui.actionHill_Climbing.setEnabled(False)
            self.ui.actionSimulated_Anneling.setEnabled(False)

    def check_clustering_buttons(self):
        if self.input_image is not None:
            self.ui.actionK_Means.setEnabled(True)
            self.ui.actionAffinity_Propagation.setEnabled(True)
            self.ui.actionMean_shift.setEnabled(True)
            self.ui.actionHierarchical_Clustering.setEnabled(True)
            self.ui.actionSpectral_Clustering.setEnabled(True)
            self.ui.actionDBSCAN.setEnabled(True)

            self.ui.toolButton_kMeans.setEnabled(True)
            self.ui.toolButton_affinityPropagation.setEnabled(True)
            self.ui.toolButton_meanShift.setEnabled(True)
            self.ui.toolButton_hierarchicalClustering.setEnabled(True)
            self.ui.toolButton_spectralClustering.setEnabled(True)
            self.ui.toolButton_dbscan.setEnabled(True)
        else:
            self.ui.actionK_Means.setEnabled(False)
            self.ui.actionAffinity_Propagation.setEnabled(False)
            self.ui.actionMean_shift.setEnabled(False)
            self.ui.actionHierarchical_Clustering.setEnabled(False)
            self.ui.actionSpectral_Clustering.setEnabled(False)
            self.ui.actionDBSCAN.setEnabled(False)

            self.ui.toolButton_kMeans.setEnabled(False)
            self.ui.toolButton_affinityPropagation.setEnabled(False)
            self.ui.toolButton_meanShift.setEnabled(False)
            self.ui.toolButton_hierarchicalClustering.setEnabled(False)
            self.ui.toolButton_spectralClustering.setEnabled(False)
            self.ui.toolButton_dbscan.setEnabled(False)


    def check_undo_redo_buttons(self):
        print("HELLO")
        # if len(self.initial_undo_stack) > 1:
        #     self.ui.toolButton_undoInitialSolution.setEnabled(True)
        #     self.ui.actionUndo_Initial_Solution.setEnabled(True)
        # else:
        #     self.ui.toolButton_undoInitialSolution.setEnabled(False)
        #     self.ui.actionUndo_Initial_Solution.setEnabled(False)
        #
        # if len(self.initial_redo_stack) > 0:
        #     self.ui.toolButton_redoInitialSolution.setEnabled(True)
        #     self.ui.actionRedo_Initial_Solution.setEnabled(True)
        # else:
        #     self.ui.toolButton_redoInitialSolution.setEnabled(False)
        #     self.ui.actionRedo_Initial_Solution.setEnabled(False)
        #
        # if len(self.final_undo_stack) > 1:
        #     self.ui.toolButton_undoFinalSolution.setEnabled(True)
        #     self.ui.actionUndo_Final_Solution.setEnabled(True)
        # else:
        #     self.ui.toolButton_undoFinalSolution.setEnabled(False)
        #     self.ui.actionUndo_Final_Solution.setEnabled(False)
        #
        # if len(self.final_redo_stack) > 0:
        #     self.ui.toolButton_redoFinalSolution.setEnabled(True)
        #     self.ui.actionRedo_Final_Solution.setEnabled(True)
        # else:
        #     self.ui.toolButton_redoFinalSolution.setEnabled(False)
        #     self.ui.actionRedo_Final_Solution.setEnabled(False)

    def update_input_image(self):
        self.input_image = io.imread("resources/temp/input.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/input.png"))

    def update_output_image(self):
        self.output = io.imread("resources/temp/output.png")
        self.ui.label_initialSolution.setPixmap(QtGui.QPixmap("resources/temp/output.png"))

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
        self.ui.toolButton_exportAsInitialSolution.setIcon(QtGui.QIcon("resources/icons/image.png"))
        self.ui.toolButton_clearInitialSolution.setIcon(QtGui.QIcon("resources/icons/clear_input.png"))
        self.ui.toolButton_undoInitialSolution.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.toolButton_redoInitialSolution.setIcon(QtGui.QIcon("resources/icons/redo.png"))

        self.ui.toolButton_saveFinalSolution.setIcon(QtGui.QIcon("resources/icons/save.png"))
        self.ui.toolButton_exportAsFinalSolution.setIcon(QtGui.QIcon("resources/icons/image.png"))
        self.ui.toolButton_clearFinalSolution.setIcon(QtGui.QIcon("resources/icons/clear_output.png"))
        self.ui.toolButton_undoFinalSolution.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.toolButton_redoFinalSolution.setIcon(QtGui.QIcon("resources/icons/redo.png"))

        self.ui.toolButton_kMeans.setIcon(QtGui.QIcon("resources/icons/km.png"))
        self.ui.toolButton_affinityPropagation.setIcon(QtGui.QIcon("resources/icons/ap.png"))
        self.ui.toolButton_meanShift.setIcon(QtGui.QIcon("resources/icons/ms.png"))
        self.ui.toolButton_spectralClustering.setIcon(QtGui.QIcon("resources/icons/sc.png"))
        self.ui.toolButton_hierarchicalClustering.setIcon(QtGui.QIcon("resources/icons/hc.png"))
        self.ui.toolButton_dbscan.setIcon(QtGui.QIcon("resources/icons/dbscan.png"))

        self.ui.toolButton_hillClimbing.setIcon(QtGui.QIcon("resources/icons/hillclimb.png"))
        self.ui.toolButton_simulatedAnneling.setIcon(QtGui.QIcon("resources/icons/simulatedanneling.png"))
        self.ui.toolButton_exit.setIcon(QtGui.QIcon("resources/icons/exit.png"))

        self.ui.actionOpen_Data.setIcon(QtGui.QIcon("resources/icons/open.png"))

        self.ui.actionSave_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/save.png"))
        self.ui.actionExport_As_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/image.png"))
        self.ui.actionClear_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/clear_input.png"))
        self.ui.actionUndo_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.actionRedo_Initial_Solution.setIcon(QtGui.QIcon("resources/icons/redo.png"))

        self.ui.actionSave_Final_Solution.setIcon(QtGui.QIcon("resources/icons/saveas.png"))
        self.ui.actionExport_As_Final_Solution.setIcon(QtGui.QIcon("resources/icons/image.png"))
        self.ui.actionClear_Final_Solution.setIcon(QtGui.QIcon("resources/icons/clear_output.png"))
        self.ui.actionUndo_Final_Solution.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.actionRedo_Final_Solution.setIcon(QtGui.QIcon("resources/icons/redo.png"))

        self.ui.actionK_Means.setIcon(QtGui.QIcon("resources/icons/km.png"))
        self.ui.actionAffinity_Propagation.setIcon(QtGui.QIcon("resources/icons/ap.png"))
        self.ui.actionMean_shift.setIcon(QtGui.QIcon("resources/icons/ms.png.png"))
        self.ui.actionSpectral_Clustering.setIcon(QtGui.QIcon("resources/icons/sc.png"))
        self.ui.actionHierarchical_Clustering.setIcon(QtGui.QIcon("resources/icons/hc.png"))
        self.ui.actionDBSCAN.setIcon(QtGui.QIcon("resources/icons/dbscan.png"))

        self.ui.actionHill_Climbing.setIcon(QtGui.QIcon("resources/icons/hillclimb.png"))
        self.ui.actionSimulated_Anneling.setIcon(QtGui.QIcon("resources/icons/simulatedanneling.png"))

        self.ui.actionExit.setIcon(QtGui.QIcon("resources/icons/exit.png"))

        self.ui.menuClear.setIcon(QtGui.QIcon("resources/icons/clear.png"))
