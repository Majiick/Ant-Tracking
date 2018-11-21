# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ants.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(146, 374)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.playButton = QtGui.QPushButton(self.centralwidget)
        self.playButton.setGeometry(QtCore.QRect(10, 10, 112, 34))
        self.playButton.setObjectName(_fromUtf8("playButton"))
        self.pauseButton = QtGui.QPushButton(self.centralwidget)
        self.pauseButton.setGeometry(QtCore.QRect(10, 50, 112, 34))
        self.pauseButton.setObjectName(_fromUtf8("pauseButton"))
        self.exitButton = QtGui.QPushButton(self.centralwidget)
        self.exitButton.setGeometry(QtCore.QRect(10, 90, 112, 34))
        self.exitButton.setObjectName(_fromUtf8("exitButton"))
        self.togglePathsButton = QtGui.QPushButton(self.centralwidget)
        self.togglePathsButton.setGeometry(QtCore.QRect(10, 160, 112, 34))
        self.togglePathsButton.setObjectName(_fromUtf8("togglePathsButton"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 140, 68, 19))
        self.label.setObjectName(_fromUtf8("label"))
        self.toggleBoxesButton = QtGui.QPushButton(self.centralwidget)
        self.toggleBoxesButton.setGeometry(QtCore.QRect(10, 200, 112, 34))
        self.toggleBoxesButton.setObjectName(_fromUtf8("toggleBoxesButton"))
        self.orignalImageButton = QtGui.QPushButton(self.centralwidget)
        self.orignalImageButton.setGeometry(QtCore.QRect(10, 240, 112, 34))
        self.orignalImageButton.setObjectName(_fromUtf8("orignalImageButton"))
        self.maskImageButton = QtGui.QPushButton(self.centralwidget)
        self.maskImageButton.setGeometry(QtCore.QRect(10, 280, 112, 34))
        self.maskImageButton.setObjectName(_fromUtf8("maskImageButton"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 146, 31))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.playButton.setText(_translate("MainWindow", "Play", None))
        self.pauseButton.setText(_translate("MainWindow", "Pause", None))
        self.exitButton.setText(_translate("MainWindow", "Exit", None))
        self.togglePathsButton.setText(_translate("MainWindow", "Toggle Paths", None))
        self.label.setText(_translate("MainWindow", "Tools:", None))
        self.toggleBoxesButton.setText(_translate("MainWindow", "Toggle Boxes", None))
        self.orignalImageButton.setText(_translate("MainWindow", "Original Image", None))
        self.maskImageButton.setText(_translate("MainWindow", "Mask Image", None))



