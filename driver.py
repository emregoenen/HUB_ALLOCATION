from PyQt5 import QtGui, QtWidgets
import numpy as np
from functools import partial
from k_means_params import Ui_Form as Ui_Form_K_Means
from affinity_params import Ui_Form as Ui_Form_Affinity
from dbscan_params import Ui_Form as Ui_Form_DBSCAN
from hierarchical_params import Ui_Form as Ui_Form_Hierarchical
from spectral_params import Ui_Form as Ui_Form_Spectral
from mean_shift_params import Ui_Form as Ui_Form_MeanShift
from clustering_v2 import ClusterKMeans


class Driver:
    def __init__(self, ui):
        self.ui = ui
        self.input_image = None
        self.output_image = None
        self.save_path = None
        self.setup_icons()
        self.setup_signal_slots()
        self.create_widgets()

    def exit(self):
        QtWidgets.QApplication.quit()

    def setup_signal_slots(self):
        self.ui.actionOpen_Data.triggered.connect(self.open_data)
        self.ui.actionSave_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionSave_Final_Solution.triggered.connect(self.say_hello)
        self.ui.actionExit.triggered.connect(self.exit)
        self.ui.actionClear_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionClear_Final_Solution.triggered.connect(self.say_hello)
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
        self.ui.actionHill_Climbing.triggered.connect(self.say_hello)
        self.ui.actionSimulated_Anneling.triggered.connect(self.say_hello)
        self.ui.actionExport_As_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionExport_As_Final_Solution.triggered.connect(self.say_hello)

    def open_data(self):
        try:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(filter="Text files (*.txt)")
            self.data = np.loadtxt(fname)
            print(self.data)
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Couldn't open file.")

    def say_hello(self):
        print("HELLO THERE !")

    def open_k_means(self):
        self.kmeans_widget.show()

    def k_means(self):
        n_clusters = int(self.kmeans_ui.n_clusters.text())
        init = self.kmeans_ui.init.currentText()
        max_iter = int(self.kmeans_ui.max_iter.text())
        algorithm = self.kmeans_ui.algorithm.currentText()
        ClusterKMeans(self.data)

    def open_affinity(self):
        self.affinity_widget.show()

    def affinity(self):
        damping = float(self.affinity_ui.damping.text())
        max_iter = int(self.affinity_ui.max_iter.text())
        convergence_iter = int(self.affinity_ui.convergence_iter.text())
        affinity = self.affinity_ui.affinity.currentText()
        random_state = int(self.affinity_ui.random_state.text())

    def open_dbscan(self):
        self.dbscan_widget.show()

    def dbscan(self):
        eps = float(self.dbscan_ui.eps.text())
        min_samples = int(self.dbscan_ui.min_samples.text())
        algorithm = self.dbscan_ui.algorithm.currentText()
        p = float(self.dbscan_ui.p.text())

    def open_hierarchical(self):
        self.hierarchical_widget.show()

    def hierarchical(self):
        n_clusters = int(self.hierarchical_ui.n_clusters.text())
        affinity = self.hierarchical_ui.affinity.currentText()
        linkage = self.hierarchical_ui.linkage.currentText()

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

    def open_spectral(self):
        self.spectral_widget.show()

    def spectral(self):
        n_clusters = int(self.spectral_ui.n_clusters.text())
        n_components = int(self.spectral_ui.n_components.text())
        n_init = int(self.spectral_ui.n_init.text())
        assign_labels = self.spectral_ui.assign_labels.currentText()

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
