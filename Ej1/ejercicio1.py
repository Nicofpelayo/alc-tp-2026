import numpy as np
import pandas as pd
import alc

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

def encontrar_m_MC(Q):
    Np = 100

    m_randoms = np.random.uniform(low=-100, high=100, size=(Np, 12))
    m_randoms = alc.normaliza(m_randoms,2)

    Qt_Q = alc.multiplicar(alc.traspuesta(Q), Q)

    min = -1
    m_min = None
    for m in m_randoms:
        mt = alc.traspuesta(m)
        Qt_Q_m = alc.multiplicar(Qt_Q, m)
        mt_Qt_Q_m = alc.multiplicar(mt, Qt_Q_m)

        if min == -1 or min > mt_Qt_Q_m:
            min = mt_Qt_Q_m
            m_min = m

    return m_min

m = encontrar_m_MC(Q)
print(m)