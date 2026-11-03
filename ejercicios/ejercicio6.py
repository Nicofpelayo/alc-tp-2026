import numpy as np
import alc
import pandas as pd
import cv2
from matplotlib import pyplot as plt
filepath_base = '../template-alumnos/heli'

matriz_K = pd.read_csv(f'{filepath_base}/K.csv', sep=';', header=None).to_numpy()
obj_pts = pd.read_csv(f'{filepath_base}/object_points.csv', sep=';', header=0, index_col=0)

#ejercicio 6.a:
#Para aplicar el algoritmo POSIT se necesita que los puntos sean coplanares;
#en este caso los 5 puntos dados se ubican en el plano UV 
filename = f'{filepath_base}/row_00248.png'
rgb = cv2.imread(filename)

plt.figure(figsize=(12,8), dpi=100)
plt.imshow(rgb)

centro =  (525,1000)
esq_sup_izq = (403,1095)
esq_sup_der = (390,947)
esq_inf_der = (640,915)
esq_inf_izq = (656,1063)

plt.plot(centro[0], centro[1],'ro', markersize=3)
plt.plot(esq_sup_izq[0], esq_sup_izq[1],'ro', markersize=3)
plt.plot(esq_sup_der[0], esq_sup_der[1],'ro', markersize=3)
plt.plot(esq_inf_der[0], esq_inf_der[1],'ro', markersize=3)
plt.plot(esq_inf_izq[0], esq_inf_izq[1],'ro', markersize=3)

plt.show()

#ejercicio 6.b:
    
def raicesCuadradasDeCCuadrado(Re, Im):
    # Buscamos λ y μ
    # ||C^2||^2 = Re^2 + Im^2
    #           = (λ^2 − μ^2)^2 + (2λμ)^2
    #           = λ^4 − 2λ^2μ^2 + μ^4 + 4λ^2μ^2
    #           = (λ^2 + μ^2)^2
    #   
    # Sea r = sqrt(Re^2 + Im^2). Tenemos
    #   r  = sqrt(Re^2 + Im^2) = sqrt((λ^2 + μ^2)^2) = λ^2 + μ^2
    # Resumiendo,
    #   r = λ^2 + μ^2
    #   Re = λ^2 − μ^2
    # Sumando y restando con Re e Im:
    #     
    #    λ^2 = (r + Re) / 2
    #    μ^2 = (r − Re) / 2
    #
    #  Si no, separamos en dos casos para no dividir por numeros chicos
    #    Si Re > 0:   λ = sqrt(((r + Re) / 2))  =>   μ = Im / (2λ)
    #    Si Re < 0:   μ = sqrt(((r − Re) / 2))  =>   λ = Im / (2μ)
    #  ya que 2λμ = Im
    #  Si r = 0 => Re = Im = 0, con lo cual λ = 0 y μ = 0

    r = np.sqrt(Re*Re + Im*Im)
    if alc.iszero(r):
        return 0, 0
    
    if Re > 0:
        lambd = np.sqrt((r + Re) / 2)
        mu = Im / (2*lambd)
    else:
        mu = np.sqrt((r - Re) / 2)
        lambd = Im / (2*mu)

    return lambd, mu


def calculoLambdaMu(I0,J0):
    # C = λ + iμ
    # C^2 = λ^2 − µ^2 + 2i λµ
    # C^2 = (λ^2 − µ^2) + i (2λµ)
    # C^2 = (||J0||^2 − ||I0||^2) + i (2 (-I0.T J0))
    # C^2 = (||J0||^2 − ||I0||^2) + i (-2 I0.T J0)

    # Calculo parte real
    norm2_I0 = alc.norma(I0, 2)
    norm2_J0 = alc.norma(J0, 2)
    norm2_I0_cuad = norm2_I0 * norm2_I0
    norm2_J0_cuad = norm2_J0 * norm2_J0
    Re = norm2_J0_cuad - norm2_I0_cuad

    # Calculo imaginaria
    Im = (-2) * alc.producto_interno(I0, J0)

    lambd, mu = raicesCuadradasDeCCuadrado(Re, Im)
    return lambd, mu

#ejercicio 6.c:
    
def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    """
    Devuelve una pose correspondiente a una 
    posible solución i, j, k, Z0, R, t de (Is, Js)
    """
    norm_I = alc.norma(Is, 2)
    norm_J = alc.norma(Js, 2)

    i = Is / norm_I
    j = Js / norm_J
    k = alc.producto_vectorial(i,j)
    Z0 = fx / norm_I

    R = np.array([i,j,k])

    # Obtenemos t
    X0 = (Z0 / fx) * m0[0] # x0
    Y0 = (Z0 / fx) * m0[1] # y0
    T = np.array([X0, Y0, Z0])
    t=T-alc.calcularAx(R, M0)
    
    return i,j,k,Z0,R,t


#ejercicio 6.d:
def RMSE(m1,m2):
    sumatoria=0
    for i in range(len(m1)):
        norma=alc.norma(m1[i]-m2[i],2)
        sumatoria+=norma**2 
    valor=np.sqrt(sumatoria/len(m1))
    return valor
                  
        
def evaluarPose(R, t, obj_vecs, img_vecs, K):
    proyecciones_validas=0 
    proyecciones=[]
    comparaciones_validas=[]
    for i in range(len(obj_vecs)):
        punto=obj_vecs[i]
        coords_camara=alc.calcularAx(R,punto) + t
        if coords_camara[2]>0:
            proyecciones_validas+=1 
            proyeccion=alc.calcularAx(K,coords_camara)
            proyeccion[0]=proyeccion[0]/proyeccion[2]
            proyeccion[1]=proyeccion[1]/proyeccion[2]
            proyecciones.append(proyeccion[:2])
            comparaciones_validas.append(img_vecs[i])
    if proyecciones_validas==0:
        return 0, np.inf
    return proyecciones_validas, RMSE(proyecciones,comparaciones_validas)

#ejercicio 6.e:
    
def construir_A(obj_vecs):
    M0=obj_vecs[0] #punto de referencia
    matriz_A=[]
    for i in range(1,len(obj_vecs)):
        vector=obj_vecs[i]-M0 
        matriz_A.append(vector)
    return np.array(matriz_A)
        
def calcular_b(coords,e,parametro="u"):
    b=[]
    for i in range(1,len(coords)):
        if parametro=="u": 
            b_i=coords[i][0]*(1+e[i])-coords[0][0]
        else:
            b_i=coords[i][1]*(1+e[i])-coords[0][1]
        b.append(b_i)
    return np.array(b)

"""def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    eps=np.zeros(len(obj_vecs))
    A=contruir_A(obj_vecs)
    """
    
    
    
    