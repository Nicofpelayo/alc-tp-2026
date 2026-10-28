# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import alc

#ejercicio 1.b:
def minimo_autovalor(D):
    autovalores=[]
    for i in range(len(D)):
        autovalores.append(D[i][i])
    minimo=autovalores[0]
    indice=0
    for i in range (len(autovalores)):
        if autovalores[i]<minimo:
            minimo=autovalores[i]
            indice=i
    return minimo,indice


#Buscamos la solución de la minimización usando la SDV
#el m en el que se obtiene el mínimo mTQTQm
def minimizacion(Q): 
    """Q=UsigmaV*, entonces Qtranpuesta por Q=V poe(sigma transpuesta por sigma) por V tranpuesta
    Q=VDVT(con D diagonal con los valores singulares al cuadrado)
    Como V es unitaria V transpuesta=V-1, entonces se tiene una diagonalización
    de Q transpuesta por Q, y como Q transpuesta por Q es simétrica y real usamos
    la función diagRH del módulo alc""" 
    V,D=alc.diagRH(alc.multiplicar(alc.traspuesta(Q),Q))
    """Si x=VTm, mTQTQM=xTDx, si despejamos m obtenemos que m=Vx
    Como xTDx=sumatoria de xi**2 por (valor singular i**2)
    siempre va a ser mayor o igual que la sumatoria de
    xi **2 por el valor singular mas chico al cuadrado;
    como la norma de x al cuadrado es 1, xTDx tiene su mínimo con el 
    último valor singular al cuadrado, que se obtiene si x=e12
    """
    _,i=minimo_autovalor(D)
    x=np.zeros(12)
    x[i]=1 
    m=alc.calcularAx(V,x)
    return m 
#ejercicio 2.a:
    
def obtenerQ3d(imgPts, objPts):
    Q = np.zeros((2 * objPts.shape[0], 12))
    for i in range(objPts.shape[0]):                                         # Para cada valor medido generamos las 2 filas correspondientes
        ncoords = np.hstack((objPts[i], 1))       # https://numpy-org.translate.goog/doc/stable/reference/generated/numpy.hstack.html?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
        Q[i * 2] = np.hstack(  (ncoords, [0, 0, 0, 0] , - 1 * imgPts[i * 2] * ncoords))
        Q[i * 2 + 1] = np.hstack(([0, 0, 0, 0], ncoords , - 1 * imgPts[i * 2 + 1] * ncoords))
    return Q


objPts = pd.read_csv('../template-alumnos/calib/3d/p_W_corners.txt', sep=',', header=None).to_numpy()
imgPts = pd.read_csv('../template-alumnos/calib/3d/detected_corners.txt', sep=r'\s+', header=None).to_numpy()               
Q=obtenerQ3d(imgPts[0],objPts)

#ejercicio 2.b:
def resolverDLT3D(Q):
    m=minimizacion(Q)
    return np.reshape(m,(3,4))

print(resolverDLT3D(Q))
M = resolverDLT3D(Q)
#ejercicio 2.c:
def devolverMatricesCalibracion3d(M):
    squareM = M[:, :3]
    lastColumnM = M[:, 3][:]
    M_inverse = alc.inversa(squareM)
    Q, U = alc.QR_con_GS(M_inverse)
    R = alc.traspuesta(Q)
    K = alc.inversa(U)
    t = alc.calcularAx(K, lastColumnM)
    return R, t, K
print(devolverMatricesCalibracion3d(M))