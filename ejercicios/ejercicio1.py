import numpy as np
import pandas as pd
import alc

def buildQ(coords, proyections):
    Q = np.zeros((2 * coords.shape[0], 12))
    for i in range(coords.shape[0]):                                         # Para cada valor medido generamos las 2 filas correspondientes
        ncoords = np.hstack((coords[i], 1))       # https://numpy-org.translate.goog/doc/stable/reference/generated/numpy.hstack.html?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
        Q[i * 2] = np.hstack(  (ncoords, [0, 0, 0, 0] , - 1 * proyections[i * 2] * ncoords))
        Q[i * 2 + 1] = np.hstack(([0, 0, 0, 0], ncoords , - 1 * proyections[i * 2 + 1] * ncoords))
    return Q

filepath_base = '../template-alumnos/calib/3d'

coords = pd.read_csv(f'{filepath_base}/p_W_corners.txt', sep=',', header=None).to_numpy()
proyections = pd.read_csv(f'{filepath_base}/detected_corners.txt', sep=r'\s+', header=None).to_numpy()                  # Test para verificar que buildQ funcione, no se obtienen las proyecciones asi
Q = buildQ(coords, proyections[0])
print(Q)
print(Q.shape)

#ejercicio 1.b:
def indice_min(v):
    minimo=v[0]
    indice=0
    for i in range(len(v)):
        if v[i]<minimo:
            minimo=v[i]
            indice=i
    return indice

def minimizacion(Q):
    U, sigma, V = alc.svd_reducida(Q)
    indice=indice_min(sigma)
    m=V[:,indice]
    return m
    
m = minimizacion(Q)
print(m)