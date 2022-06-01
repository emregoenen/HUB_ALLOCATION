from PyQt5 import QtGui, QtWidgets
import numpy as np
from functools import partial


class Driver:
    def __init__(self, ui):
        self.ui = ui
        self.input_image = None
        self.output_image = None
        self.save_path = None
        self.setup_icons()
        self.setup_signal_slots()

    def exit(self):
        QtWidgets.QApplication.quit()

    def setup_signal_slots(self):
        self.ui.actionOpen_Data.triggered.connect(self.say_hello)
        self.ui.actionSave_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionSave_Final_Solution.triggered.connect(self.say_hello)
        self.ui.actionExit.triggered.connect(self.exit)
        self.ui.actionClear_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionClear_Final_Solution.triggered.connect(self.say_hello)
        self.ui.actionUndo_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionUndo_Final_Solution.triggered.connect(self.say_hello)
        self.ui.actionRedo_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionRedo_Final_Solution.triggered.connect(self.say_hello)
        self.ui.actionK_Means.triggered.connect(self.say_hello)
        self.ui.actionAffinity_Propagation.triggered.connect(self.say_hello)
        self.ui.actionMean_shift.triggered.connect(self.say_hello)
        self.ui.actionSpectral_Clustering.triggered.connect(self.say_hello)
        self.ui.actionHierarchical_Clustering.triggered.connect(self.say_hello)
        self.ui.actionDBSCAN.triggered.connect(self.say_hello)
        self.ui.actionHill_Climbing.triggered.connect(self.say_hello)
        self.ui.actionSimulated_Anneling.triggered.connect(self.say_hello)
        self.ui.actionExport_As_Initial_Solution.triggered.connect(self.say_hello)
        self.ui.actionExport_As_Final_Solution.triggered.connect(self.say_hello)


    def say_hello(self):
        print("HELLO THERE !")

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
