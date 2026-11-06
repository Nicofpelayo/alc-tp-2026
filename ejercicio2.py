# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import alc

#ejercicio 2.a:
    
def obtenerQ3d(imgPts, objPts):
    Q = np.zeros((2 * objPts.shape[0], 12))
    for i in range(objPts.shape[0]):                                         # Para cada valor medido generamos las 2 filas correspondientes
        ncoords = np.hstack((objPts[i], 1))       # https://numpy-org.translate.goog/doc/stable/reference/generated/numpy.hstack.html?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
        Q[i * 2] = np.hstack(  (ncoords, [0, 0, 0, 0] , - 1 * imgPts[i * 2] * ncoords))
        Q[i * 2 + 1] = np.hstack(([0, 0, 0, 0], ncoords , - 1 * imgPts[i * 2 + 1] * ncoords))
    return Q

filepath_base = 'calib/3d'

objPts = pd.read_csv(f'{filepath_base}/p_W_corners.txt', sep=',', header=None).to_numpy()
imgPts = pd.read_csv(f'{filepath_base}/detected_corners.txt', sep=r'\s+', header=None).to_numpy()               
Q=obtenerQ3d(imgPts[0],objPts)


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


#ejercicio 2.b:
def resolverDLT3D(Q):
    m=minimizacion(Q)
    return np.reshape(m,(3,4))


def determinante_3x3(A):
    #recorrido por primera fila
    valor= (A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1])
          - A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0])
          + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0]))
    return valor
#ejercicio 2.c y 2.d:
def signo(n):
    if n<0:
        return -1
    elif n>0:
        return 1
    return 0

def devolverMatricesCalibracion3d(M,opcional=0):
    matriz_KR = M[:,:3]
    ultima_columna_M = M[:,3]
    R_traspuesta, inv_K = alc.QR_con_GS(alc.inversa(matriz_KR))
    R = alc.traspuesta(R_traspuesta)
    K = alc.inversa(inv_K)
    t = alc.calcularAx(inv_K, ultima_columna_M)
    D = np.zeros((3, 3))
    s1 = signo(K[0, 0] * K[2, 2])
    s2 = signo(K[1, 1] * K[2, 2])
    sR = signo(determinante_3x3(R))
    d3 = s1 * s2 * sR
    d1 = s1 * d3
    d2 = s2 * d3
    D[0, 0] = d1
    D[1, 1] = d2
    D[2, 2] = d3
    R = alc.multiplicar(D, R)
    K = alc.multiplicar(K, D)
    t = alc.calcularAx(D, t) 
    K = K / K[2, 2]
    # Ejercicio 2.d 
    if opcional == 1 or opcional == 3: 
        promedio = (K[0,0] + K[1,1]) / 2 
        K[0,0] = promedio  
        K[1,1] = promedio 
    if opcional == 2 or opcional == 3: 
        K[0,1] = 0 
    return R, t, K
    
M=resolverDLT3D(Q)
#print(devolverMatricesCalibracion3d(M,opcional=0))
   
#con opcional 1,2,3 no se reconstruye M

