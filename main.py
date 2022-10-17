import os
import sys
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from display import DispApp
from PyQt5.QtWidgets import QApplication

# code for this function came from https://www.pythonpool.com/matplotlib-draw-rectangle/
def addtograph(ax,x1,x2,y1,y2,z1,z2):
    Z = np.array([[x1, y1, z1],
                    [x2, y1, z1],
                    [x2, y2, z1],
                    [x1, y2, z1],
                    [x1, y1, z2],
                    [x2, y1, z2],
                    [x2, y2, z2],
                    [x1, y2, z2]])
    verts = [[Z[0],Z[1],Z[2],Z[3]],
    [Z[4],Z[5],Z[6],Z[7]],
    [Z[0],Z[1],Z[5],Z[4]],
    [Z[2],Z[3],Z[7],Z[6]],
    [Z[1],Z[2],Z[6],Z[5]],
    [Z[4],Z[7],Z[3],Z[0]]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='c', alpha=.10))

def display(active, r, which, min_x, min_y, min_z, max_x, max_y, max_z, dminx, dminy, dminz, dmaxx, dmaxy, dmaxz):
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(projection='3d')

    for k in active:
        kk = k
        x0 = kk % (2 ** (r+1))
        kk = int(kk / (2 ** (r+1)))
        y0 = kk % (2 ** (r+1))
        kk = int(kk / (2 ** (r+1)))
        z0 = kk % (2 ** (r+1))

        x1 = min_x[which] + (max_x[which]-min_x[which]) * x0 / (2 ** (r+1))
        x2 = min_x[which] + (max_x[which]-min_x[which]) * (x0+1) / (2 ** (r+1))
        y1 = min_y[which] + (max_y[which]-min_y[which]) * y0 / (2 ** (r+1))
        y2 = min_y[which] + (max_y[which]-min_y[which]) * (y0+1) / (2 ** (r+1))
        z1 = min_z[which] + (max_z[which]-min_z[which]) * z0 / (2 ** (r+1))
        z2 = min_z[which] + (max_z[which]-min_z[which]) * (z0+1) / (2 ** (r+1))
        addtograph(ax,x1,x2,y1,y2,z1,z2)

    ax.set_xlim(dminx[which],dmaxx[which])
    ax.set_ylim(dminy[which],dmaxy[which])
    ax.set_zlim(dminz[which],dmaxz[which])
    plt.savefig(f"./images/{name[which]}_{r:04d}.png")
    #plt.show()

name = ["chaotic", "rossler", "lorenz", "aizawa", "halvorsen"]
if len(sys.argv) == 2:
    which = int(sys.argv[1])
else:
    for i in range(len(name)):
        print(i,'-',name[i])
    which = int(input('system = '))

os.environ["PYOPENCL_CTX"] = "0"
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

prg = cl.Program(ctx, open('kernels.cl', mode='rt').read()).build()

knl = prg.dostep
min_x = np.float32([-2.0, -30.0, -50.0, -4.0, -40.0])
max_x = np.float32([2.0, 30.0, 50.0, 4.0, 40.0])
min_y = np.float32([-2.0, -30.0, -50.0, -4.0, -40.0])
max_y = np.float32([2.0, 30.0, 50.0, 4.0, 40.0])
min_z = np.float32([-2.0, -30.0, -50.0, -4.0, -40.0])
max_z = np.float32([2.0, 30.0, 50.0, 4.0, 40.0])

dminx = [-2.0, -15.0, -30.0, -2.0, -15.0]
dmaxx = [2.0, 15.0, 30.0, 2.0, 15.0]
dminy = [-2.0, -15.0, -30.0, -2.0, -15.0]
dmaxy = [2.0, 15.0, 30.0, 2.0, 15.0]
dminz = [-2.0, -15.0, -10.0, -2.0, -15.0]
dmaxz = [2.0, 15.0, 40.0, 2.0, 15.0]

numSteps = [0, 100000, 100000, 100000, 100000]
ss = [0, 0.001, 0.001, 0.001, 0.001]

dim = 3
test_points = 30

r = 3
active = [i for i in range((2 ** r) ** dim)]
num_rounds = 15
app = QApplication(["main.py"])
window = DispApp(ctx)
mydisplay = True
while r <= num_rounds:
    print('r =', r, ', Number of active boxes =', len(active))

    numbatches = int(math.ceil(len(active) / 1000))
    newactive = {}
    numtestpointsall = 0
    print('Number of batches =', numbatches, '[', end='', flush=True)
    for bn in range(numbatches):
        active_batch = np.int64(active[bn * 1000 : (bn + 1) * 1000])

        numtestpoints = len(active_batch) * (test_points ** dim)
        numtestpointsall += numtestpoints

        result = np.zeros(numtestpoints, dtype=np.int64)
        d_active = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=active_batch)
        d_result = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result)

        knl.set_scalar_arg_dtypes( [None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.int32, np.int32, np.float32, np.int32, np.int32] )
        knl(queue, result.shape, None, d_active, d_result, min_x[which], max_x[which], min_y[which], max_y[which], min_z[which], max_z[which], np.int32(which), np.int32(numSteps[which]), np.float32(ss[which]), np.int32(r), np.int32(test_points))

        cl.enqueue_copy(queue, result, d_result)

        for i in range(result.shape[0]):
            if result[i] != -1:
                newactive[result[i]] = True

        print('*' + str(int(bn/10)) + '*' if bn%10 == 0 else str(bn%10), end='', flush=True)
    print(']', flush=True)
    print('Number of testing points =', numtestpointsall)

    active = []
    for k in newactive:
        active.append(k)

    if mydisplay:
        x = []
        y = []
        z = []
        for k in active:
            kk = k
            x0 = kk % (2 ** (r+1))
            kk = int(kk / (2 ** (r+1)))
            y0 = kk % (2 ** (r+1))
            kk = int(kk / (2 ** (r+1)))
            z0 = kk % (2 ** (r+1))

            x.append(min_x[which] + (max_x[which]-min_x[which]) * (x0+0.5) / (2 ** (r+1)))
            y.append(min_y[which] + (max_y[which]-min_y[which]) * (y0+0.5) / (2 ** (r+1)))
            z.append(min_z[which] + (max_z[which]-min_z[which]) * (z0+0.5) / (2 ** (r+1)))
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = np.array(z, dtype=np.float32)

        d_x = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
        d_y = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
        d_z = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=z)

        window.setall(x, d_x, d_y, d_z, queue, ctx, mf)
        window.show()
        app.exec()
    else:
        if r == 5 or r == 8 or r == 11:
            display(active, r, which, min_x, min_y, min_z, max_x, max_y, max_z, dminx, dminy, dminz, dmaxx, dmaxy, dmaxz)

    print('Number of active boxes for next step =', len(active))
    r += 1
