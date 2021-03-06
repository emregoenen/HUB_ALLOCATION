from PyQt5 import QtCore, QtGui, QtWidgets

## UI for asking DBSCAN Clustering Parameters
#  Generated from PyQT5 Designer
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(520, 463)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.eps_text = QtWidgets.QLabel(self.groupBox)
        self.eps_text.setObjectName("eps_text")
        self.horizontalLayout.addWidget(self.eps_text)
        self.eps = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.eps.sizePolicy().hasHeightForWidth())
        self.eps.setSizePolicy(sizePolicy)
        self.eps.setClearButtonEnabled(False)
        self.eps.setObjectName("eps")
        self.horizontalLayout.addWidget(self.eps)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.min_samples_text = QtWidgets.QLabel(self.groupBox_2)
        self.min_samples_text.setObjectName("min_samples_text")
        self.horizontalLayout_2.addWidget(self.min_samples_text)
        self.min_samples = QtWidgets.QLineEdit(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.min_samples.sizePolicy().hasHeightForWidth())
        self.min_samples.setSizePolicy(sizePolicy)
        self.min_samples.setClearButtonEnabled(False)
        self.min_samples.setObjectName("min_samples")
        self.horizontalLayout_2.addWidget(self.min_samples)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_6 = QtWidgets.QGroupBox(Form)
        self.groupBox_6.setTitle("")
        self.groupBox_6.setObjectName("groupBox_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.algorithm_text = QtWidgets.QLabel(self.groupBox_6)
        self.algorithm_text.setObjectName("algorithm_text")
        self.horizontalLayout_6.addWidget(self.algorithm_text)
        self.algorithm = QtWidgets.QComboBox(self.groupBox_6)
        self.algorithm.setObjectName("algorithm")
        self.algorithm.addItem("")
        self.algorithm.addItem("")
        self.algorithm.addItem("")
        self.algorithm.addItem("")
        self.horizontalLayout_6.addWidget(self.algorithm)
        self.verticalLayout.addWidget(self.groupBox_6)
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.p_text = QtWidgets.QLabel(self.groupBox_3)
        self.p_text.setObjectName("p_text")
        self.horizontalLayout_3.addWidget(self.p_text)
        self.p = QtWidgets.QLineEdit(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.p.sizePolicy().hasHeightForWidth())
        self.p.setSizePolicy(sizePolicy)
        self.p.setClearButtonEnabled(False)
        self.p.setObjectName("p")
        self.horizontalLayout_3.addWidget(self.p)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.OKButton = QtWidgets.QPushButton(self.groupBox_5)
        self.OKButton.setObjectName("OKButton")
        self.horizontalLayout_5.addWidget(self.OKButton)
        self.CancelButton = QtWidgets.QPushButton(self.groupBox_5)
        self.CancelButton.setObjectName("CancelButton")
        self.horizontalLayout_5.addWidget(self.CancelButton)
        self.verticalLayout.addWidget(self.groupBox_5)

        self.retranslateUi(Form)
        self.CancelButton.clicked['bool'].connect(Form.close)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "DBSCAN Clustering Parameters"))
        self.eps_text.setText(_translate("Form", "eps: float , default = 0.5"))
        self.eps.setText(_translate("Form", "0.5"))
        self.min_samples_text.setText(_translate("Form", "min_samples: int , default = 5"))
        self.min_samples.setText(_translate("Form", "5"))
        self.algorithm_text.setText(_translate("Form", "algorithm: str , default = auto"))
        self.algorithm.setItemText(0, _translate("Form", "auto"))
        self.algorithm.setItemText(1, _translate("Form", "ball_tree"))
        self.algorithm.setItemText(2, _translate("Form", "kd_tree"))
        self.algorithm.setItemText(3, _translate("Form", "brute"))
        self.p_text.setText(_translate("Form", "p: float , default = 2"))
        self.p.setText(_translate("Form", "2"))
        self.OKButton.setText(_translate("Form", "OK"))
        self.CancelButton.setText(_translate("Form", "Cancel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
