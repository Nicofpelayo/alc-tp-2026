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

def resolverDLT2D(imgPts, objPts):
    """
    Calcula la homografía H correspondientes a puntos del plano (objPts) y de la imagen (imgPts) mediante el algoritmo DLT 2D.
    """
    #Q de tamaño 2Nx9
    Q = obtenerQ2d(imgPts, objPts)

    h = alc.minimo_cuadrado_homogeneo(Q) 
    
    H = h.reshape(3,3) 
    return H

def v_ij(H, i, j):
    """
    Función auxiliar que calcula el vector de 6 elementos a partir 
    de las columnas i y j de la homografía H.
    """
    # (c_i)^T B c_j = B_{11} (c_{i1}c_{j1}) + B_{12}(c_{i1}c_{j2}) + B_{13}(c_{i1}c_{j3})
    #                 + B_{21} (c_{i2}c_{j1}) + B_{22}(c_{i2}c_{j2}) + B_{23}(c_{i2}c_{j3})
    #                 + B_{31} (c_{i3}c_{j1}) + B_{32}(c_{i3}c_{j2}) + B_{33}(c_{i3}c_{j3})
    # 
    # B simétrica, entonces B_{ij} = B_{ji}. Entonces
    # 
    # (c_i)^T B c_j =   B_{11} c_{i1}c_{j1}
    #                 + B_{12} (c_{i1}c_{j2} + c_{i2}c_{j1})
    #                 + B_{22} (c_{i2}c_{j2})
    #                 + B_{13} (c_{i3}c_{j1} + c_{i1}c_{j3})
    #                 + B_{23} (c_{i3}c_{j2} + c_{i2}c_{j3})
    #                 + B_{33} (c_{i3}c_{j3})
    #               = v_ij b

    c_i = H[:, i]
    c_j = H[:, j]
    return np.array([
        c_i[0] * c_j[0],
        c_i[0] * c_j[1] + c_i[1] * c_j[0],
        c_i[1] * c_j[1],
        c_i[2] * c_j[0] + c_i[0] * c_j[2],
        c_i[2] * c_j[1] + c_i[1] * c_j[2],
        c_i[2] * c_j[2]
    ])

def armarA(Hlst):
    """
    Hlst una lista de matrices correspondientes a las N vistas,
    obtenidas a partir de la resolución del sistema planteado en la ecuación 11 (i.e. resolverDLT2D).
    Devuelve la matriz A (la calibracion) de dimensiones 2N x 6 requerida por el algoritmo de Zhang.
    """
    N = len(Hlst)
    A = np.zeros((2 * N, 6))
    
    for k in range(N):
        H = Hlst[k]
        
        # Calculamos los v_ij requeridas por el paso e) del algoritmo de Zhang
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
    b = alc.minimo_cuadrado_homogeneo(A)

    B = b.reshape(3,3)

    if not alc.esSDP(B):
        B = (-1) * B

    L, _, _ = alc.calculaLDV(B)

    K = alc.inversa(alc.traspuesta(L))

    # Como B se conoce solo salvo escala, se normaliza K dividiendola
    # por K_33 para que K_33 = 1
    K = K / K[2,2]

    return K

def obtenerPose2D(Hj, K):
    """
    Devuelve la pose de la vista j como Rj y tj
    a partir de la matriz K y la Hj correspondiente.
    """
    K_inv = alc.inversa(K)
    s = alc.norma(alc.inversa(K),2)
    lambd = 1 / s

    r1 = lambd * K_inv * Hj[:,1]
    r2 = lambd * K_inv * Hj[:,2]
    r3 = alc.producto_vectorial(r1,r2)
    tj = lambd * K_inv * Hj[:,3]

    Rj = np.hstack((
        r1.reshape((-1,1)),
        r2.reshape((-1,1)),
        r3.reshape((-1,1))))

    return Rj, tj