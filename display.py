import os
import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QDial, QGridLayout, QLabel, QPushButton, QCheckBox
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import pyopencl as cl
import math


class DispApp(QMainWindow):
    def __init__(self, ctx):
        super().__init__()

        prg = cl.Program(ctx, open('kernels_disp.cl', mode='rt').read()).build()
        self.knl = prg.display

        self.setWindowTitle("Widgets App")

        layout = QGridLayout()
        self.widgets = [
            [QLabel(),200,10, 0,0, 1,1],
            [QDial(),200,200, 1,0, 1,1],
            [QLabel(),200,10, 2,0, 1,1],
            [QDial(),200,200, 3,0, 1,1],
            [QLabel(),200,10, 4,0, 1,1],
            [QDial(),200,200, 5,0, 1,1],
            [QCheckBox('Log colors'),200,10, 6,0, 1,1],
            [QCheckBox('Stereo glasses'),200,10, 7,0, 1,1],
            [QPushButton('Save picture'),200,10, 8,0, 1,1],
            [QPushButton('Exit'),200,10, 9,0, 1,1],
            [QLabel(),1500,1500, 0,1, 10,1],
        ]

        for i in range(len(self.widgets)):
            w = self.widgets[i]
            cw = w[0]
            cw.resize(w[1],w[2])
            if i == 0:
                cw.setText('Rotation 1')
            if i == 1:
                cw.setMinimum(-180)
                cw.setMaximum(180)
                cw.setValue(0)
                cw.sliderReleased.connect(lambda:self.released())
            if i == 2:
                cw.setText('Rotation 2')
            if i == 3:
                cw.setMinimum(-180)
                cw.setMaximum(180)
                cw.setValue(0)
                cw.sliderReleased.connect(lambda:self.released())
            if i == 4:
                cw.setText('Zoom')
            if i == 5:
                cw.setMinimum(1)
                cw.setMaximum(300)
                cw.setValue(10)
                cw.sliderReleased.connect(lambda:self.released())
            if i == 6:
                cw.stateChanged.connect(lambda:self.released())
            if i == 7:
                cw.stateChanged.connect(lambda:self.released())
            if i == 8:
                cw.clicked.connect(lambda:self.movie())
            if i == 9:
                cw.clicked.connect(lambda:self.pressed())
            layout.addWidget(cw, w[3], w[4], w[5], w[6])

        widget = QWidget()
        widget.setLayout(layout)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)

    def movie(self):
        self.redraw('attractor.png')

    def setall(self, x, d_x, d_y, d_z, queue, ctx, mf):
        self.x = x
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.ctx = ctx
        self.mf = mf
        self.queue = queue

        self.redraw()

    def redraw(self, fn=None):
        image = np.zeros((1500*1500*4), dtype=np.int32)
        d_image = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=image)
        imw = 1500
        imh = 1500
        al = self.widgets[1][0].value() * 2.0 * math.pi / 360.0
        be = self.widgets[3][0].value() * 2.0 * math.pi / 360.0
        mu = self.widgets[5][0].value()
        self.knl.set_scalar_arg_dtypes( [None, None, None, None, np.int32, np.int32, np.float32, np.float32, np.float32, np.int32])
        self.knl(self.queue, self.x.shape, None, self.d_x, self.d_y, self.d_z, d_image, np.int32(imw), np.int32(imh), np.float32(al),np.float32(be),np.float32(mu), np.int32(self.widgets[7][0].checkState()))

        cl.enqueue_copy(self.queue, image, d_image)
        im2 = np.array(image, dtype=np.float64)
        if self.widgets[6][0].checkState():
            im2 = np.log(1.0 + 10.0 * im2 / max(im2))
        im3 = np.round(65535.0-(65535.0 * im2 / max(im2)))
        im4 = np.array(im3, dtype=np.uint16)
        im4.reshape((1500,1500,4))

        qim = QImage(im4.data, imw, imh, QImage.Format_RGBA64)
        if fn != None:
            qim.save(fn,'PNG')
        pix = QPixmap.fromImage(qim)
        self.widgets[10][0].setPixmap(pix)

    def released(self):
        self.redraw()

    def pressed(self):
        exit(1)