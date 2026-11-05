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
        u_i = imgPts[i,0]
        v_i = imgPts[i,1]

        Q[i * 2] = np.hstack((P_i, [0, 0, 0], -1 * u_i * P_i))
        Q[i * 2 + 1] = np.hstack(([0, 0, 0], P_i, -1 * v_i * P_i))
    return Q

def obtenerTHartley(pts):
    # 1. Hay una traslación de los puntos de manera que 
    # su centroide cae en el origen de coordenadas
    # 
    # 2. Todos los puntos son normalizados de manera que
    # el promedio de sus distancias al origen sqrt(2).
    
    # Calculo de centroide (traslacion)
    xs = pts[:,0]
    ys = pts[:,1]
    cx = np.mean(xs)
    cy = np.mean(ys)
    c = np.array([cx, cy])
    
    # Promedio de distancias (escala)
    distancias = [alc.norma(pts[i] - c,2) for i in range(len(pts))]
    dist_promedio = np.mean(distancias)
    s = np.sqrt(2) / dist_promedio #escala

    #      [1 0 −c_x]         [s 0 0]
    # Tr = [0 1 −c_y],   S =  [0 s 0]
    #      [0 0   1 ]         [0 0 1]
    #
    #            [1 0 −c_x] [s 0 0]   [s 0 -s·c_x]
    # T = S·TR = [0 1 −c_y]·[0 s 0] = [0 s -s·c_y]
    #            [0 0   1 ] [0 0 1]   [0 0    1  ]
    #
    #                                [1 0 c_x] [1/s  0  0]   [1/s  0   c_x]
    # T^{-1} = Tr^{-1} · S^{-1}  =   [0 1 c_y]·[ 0  1/s 0] = [ 0  1/s  c_y]
    #                                [0 0  1 ] [ 0   0  1]   [ 0   0    1 ]
    #
    # Tr es una traslacion, con lo cual la inversa es trasladarlo de nuevo a donde estaba (i.e. invertir los signos)
    # S es una escala => la inversa es dividir por lo que se escalo

    T = np.array(
        [[s, 0, -s*cx],
         [0, s, -s*cy],
         [0, 0, 1]])

    T_inv = np.array(
            [[1/s,  0 , cx],
             [ 0 , 1/s, cy],
             [ 0 ,  0 ,  1]])

    return T, T_inv

def resolverDLT2D_PtsNormalizados(imgPts, objPts):
    #Q de tamaño 2Nx9
    Q = obtenerQ2d(imgPts, objPts)
    h = alc.minimo_cuadrado_homogeneo(Q) 
    
    return h.reshape(3,3) 

def normalizarPts(T, pts):
    normalizados = []

    for i in range(pts.shape[0]):
        p_i = np.hstack((pts[i], 1))
        normalizados.append(alc.multiplicar(T, p_i))

    return np.array(normalizados)[:,:-1]

def resolverDLT2D(imgPts, objPts):
    """
    Calcula la homografía H correspondientes a puntos del plano (objPts) y de la imagen (imgPts) mediante el algoritmo DLT 2D.
    """
    
    # Normalizar usando Hartley
    Timg, Timg_inv = obtenerTHartley(imgPts)
    Tworld, _ = obtenerTHartley(objPts)

    
    imgPtsNormalizados = normalizarPts(Timg, imgPts)
    objPtsNormalizados = normalizarPts(Tworld, objPts)

    H_hat = resolverDLT2D_PtsNormalizados(imgPtsNormalizados, objPtsNormalizados)

    # H = Timg^-1 H_hat Tworld
    return alc.multiplicar(Timg_inv, alc.multiplicar(H_hat, Tworld))

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

    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    if not alc.esSDP(B):
        B = (-1) * B

    L = alc.calculaCholesky(B)

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
    c1, c2, c3 = Hj[:, 0], Hj[:, 1], Hj[:, 2]
    K_inv_c1 = alc.multiplicar(K_inv, c1)
    K_inv_c2 = alc.multiplicar(K_inv, c2)
    K_inv_c3 = alc.multiplicar(K_inv, c3)

    s = alc.norma(K_inv_c1, 2)
    lambd = 1 / s

    r1 = lambd * K_inv_c1
    r2 = lambd * K_inv_c2
    r3 = alc.producto_vectorial(r1,r2)
    tj = lambd * K_inv_c3       
    
    # t =  λK^-1·c3
    # Si t_z < 0, el punto quedó detrás de la camara,
    #   (Recordar que H vino de resolver min_{||h||_2=1} Qh = 0),
    #   con lo cual H puede ser H o -H
    # Invertimos los signos para ajustar
    #   (r3 queda igual)
    if tj[2] < 0:
        r1 = r1 * (-1)
        r2 = r2 * (-1)
        tj = tj * (-1)

    Rj = np.hstack((
        r1.reshape((-1,1)),
        r2.reshape((-1,1)),
        r3.reshape((-1,1))))

    # se aplican restricciones a la matriz de rotación
    # ortonormalizamos con Gram Schmidt
    Rj, _ = alc.QR_con_GS(Rj)

    return Rj, tj