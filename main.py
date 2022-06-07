import sys
from ui import Ui_MainWindow
from PyQt5 import QtWidgets
from driver import Driver


app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
driver = Driver(ui, MainWindow)
MainWindow.show()
sys.exit(app.exec_())