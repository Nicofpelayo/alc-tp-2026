import numpy as np
import pandas as pd
def buildQ(coords, proyections):
    print(proyections)
    Q = np.zeros((2 * coords.shape[0], 12))
    for i in range(coords.shape[0]):                                         # Para cada valor medido generamos las 2 filas correspondientes
        ncoords = np.hstack((coords[i], 1))       # https://numpy-org.translate.goog/doc/stable/reference/generated/numpy.hstack.html?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
        Q[i * 2] = np.hstack(  (ncoords, [0, 0, 0, 0] , - 1 * proyections[i * 2] * ncoords))
        Q[i * 2 + 1] = np.hstack(([0, 0, 0, 0], ncoords , - 1 * proyections[i * 2 + 1] * ncoords))
    return Q


coords = pd.read_csv('template-alumnos/calib/3d/p_W_corners.txt', sep=',', header=None).to_numpy()
proyections = pd.read_csv('template-alumnos/calib/3d/detected_corners.txt', sep=' ', header=None).to_numpy()                  # Test para verificar que buildQ funcione, no se obtienen las proyecciones asi
Q = buildQ(coords, proyections[0])
print(Q)
print(Q.shape)