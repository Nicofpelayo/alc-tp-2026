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


coords = pd.read_csv('../template-alumnos/calib/3d/p_W_corners.txt', sep=',', header=None).to_numpy()
proyections = pd.read_csv('../template-alumnos/calib/3d/detected_corners.txt', sep=r'\s+', header=None).to_numpy()                  # Test para verificar que buildQ funcione, no se obtienen las proyecciones asi
Q = buildQ(coords, proyections[0])
print(Q)
print(Q.shape)
"""
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
"""
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

m=minimizacion(Q)
print(m)
alc.norma(m,2)
#cambie la tolerancia de esSimetrica a 1e-7 para que pase , hay que chequear


