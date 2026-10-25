import numpy as np
import pandas as pd
import alc

def obtenerQ2d(imgPts, objPts):
    """
    Devuelve la matriz Q a partir de la ecuación 11
    """
    n = objPts.shape[0]
    Q = np.zeros((2 * n, 9))
    for i in range(n):                                         # Para cada valor medido generamos las 2 filas correspondientes
        P_i = np.hstack((objPts[i], 1))       # https://numpy-org.translate.goog/doc/stable/reference/generated/numpy.hstack.html?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
        u_i = imgPts[i*2]
        v_i = imgPts[i*2 + 1]

        Q[i * 2] = np.hstack((P_i, [0, 0, 0], -1 * u_i * P_i))
        Q[i * 2 + 1] = np.hstack(([0, 0, 0], P_i, -1 * v_i * P_i))
    return Q


#coords = pd.read_csv('template-alumnos/calib/coplanar/p_W_corners.txt', sep=',', header=None).to_numpy()
#proyections = pd.read_csv('template-alumnos/calib/coplanar/detected_corners.txt', sep=' ', header=None).to_numpy()                  # Test para verificar que buildQ funcione, no se obtienen las proyecciones asi
#Q = buildQ(coords, proyections[0])
#print(Q)
#print(Q.shape)

def resolverDLT2D(imgPts, objPts):
    """
    Calcula la homografía H mediante el algoritmo DLT 2D a partir de  puntos correspondientes en el plano (objPts) y en la imagen (imgPts).
    Resuelve el sistema homogéneo Q * h = 0 usando SVD.
    """
    #Armamos la matriz Q de 2Nx9 
    Q = obtenerQ2d(imgPts, objPts)

    #Resolvemos el sistema Q * h = 0 usando SVD reducida que con k=1 devuelve el ultimo
    #vector singular de V (el correspondiente al menor valor singular)
    _, _, hatV = alc.svd_reducida(Q, k=1)

    h = hatV[:,0] # Como hatV nos devuelve los vectores columna directamente, tomamos toda la primera columna (índice 0).
    H = h.reshape(3,3) # Reshapeamos el vector h en una matriz 3x3
    H = H/H[2,2] # Normalizamos la matriz H dividiendo por el elemento H[2,2]
    return H

def v_ij(H, i, j):
    """
    Función auxiliar que calcula el vector de 6 elementos a partir 
    de las columnas i y j de la homografía H.
    """
    h1 = H[:, i]
    h2 = H[:, j]
    return np.array([
        h1[0] * h2[0],
        h1[0] * h2[1] + h1[1] * h2[0],
        h1[1] * h2[1],
        h1[2] * h2[0] + h1[0] * h2[2],
        h1[2] * h2[1] + h1[1] * h2[2],
        h1[2] * h2[2]
    ])

def armarA(Hlst):
    """
    Hlst una lista de matrices correspondientes a las N capturas de las miras,
    obtenidas a partir de la resolución del sistema planteado en la ecuación 11.
    Devuelve la matriz A (la calibracion?) de dimensiones 2N x 6, donde N es la cantidad de capturas de las miras.
    """
    N = len(Hlst)
    A = np.zeros((2 * N, 6))
    
    for k in range(N):
        H = Hlst[k]
        
        # Función auxiliar para cada combinación requerida por Zhang
        v12 = v_ij(H, 0, 1)
        v11 = v_ij(H, 0, 0)
        v22 = v_ij(H, 1, 1)
        
        # Asignamos las dos filas correspondientes en la matriz A
        A[2 * k]     = v12
        A[2 * k + 1] = v11 - v22
        
    return A

def calcularMatrizK2D(A):
    """
    Recibe la matriz A y calcula la matriz K
    """
    pass

def obtenerPose2D(Hj, K):
    """
    Devuelve la pose de la vista j como Rj y tj
    a partir de la matriz K y la Hj correspondiente.
    """
    pass