import numpy as np
import alc
import pandas as pd
import cv2
from matplotlib import pyplot as plt
filepath_base = '../template-alumnos/heli'

matriz_K = pd.read_csv(f'{filepath_base}/K.csv', sep=';', header=None).to_numpy()
puntos_3d = pd.read_csv(f'{filepath_base}/object_points.csv', sep=';', header=0, index_col=0).to_numpy(dtype=float)

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
        
def centrar_coords(img_vecs,K):
    c_u = K[0][2]
    c_v = K[1][2]
    coords_centradas = []
    for i in range(len(img_vecs)):
        x = img_vecs[i][0] - c_u
        y = img_vecs[i][1] - c_v
        coords_centradas.append(np.array([x,y]))
    return coords_centradas

def calcular_b(coords,e,parametro="u"):
    b=[]
    for i in range(1,len(coords)):
        if parametro=="u": 
            b_i=coords[i][0]*(1+e[i])-coords[0][0]
        else:
            b_i=coords[i][1]*(1+e[i])-coords[0][1]
        b.append(b_i)
    return np.array(b)

def mejor_pose_camara(pose1,pose2,n):
    mejor_pose = None
    num=None
    if pose1[0] ==n and pose2[0]==n:
        if pose1[1]<pose2[1]:
            mejor_pose=pose1
            num=1
        else:
            mejor_pose=pose2 
            num=2
    elif pose1[0] ==n:
        mejor_pose=pose1
        num=1
    elif pose2[0]==n:
        mejor_pose=pose2 
        num=2
    return mejor_pose, num

def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    eps = np.zeros(len(obj_vecs))
    A = construir_A(obj_vecs)
    coords = centrar_coords(img_vecs, K)
    pseudo_inv_A = alc.pseudoinversa(A)
    Nu = alc.nucleo(A)
    n = Nu[:, 0]
    n = n / alc.norma(n, 2)
    fx = K[0, 0]
    m0 = coords[0]
    M0 = obj_vecs[0]
    for j in range(MaxIt):
        b_u = calcular_b(coords, eps, "u")
        b_v = calcular_b(coords, eps, "v")
        I0 = alc.calcularAx(pseudo_inv_A, b_u)
        J0 = alc.calcularAx(pseudo_inv_A, b_v)
        #sol1:
        lambda1,mu1 = calculoLambdaMu(I0, J0)
        #sol2: cambiamos los signos, para obtener la segunda raíz compleja
        lambda2,mu2 = -lambda1,-mu1
        #calculamos I=I0+lambda*n, J=J0+mu*n para cada lambda, mu
        Is = [I0+lambda1*n,I0+lambda2*n]
        Js = [J0+mu1*n,J0+mu2*n]
        i1, j1, k1, Z01, R1, t1 = calcularSalidaPOSIT(Is[0], Js[0], fx, m0, M0)
        i2, j2, k2, Z02, R2, t2 = calcularSalidaPOSIT(Is[1], Js[1], fx, m0, M0)
        #evaluamos las poses:
        pose1=evaluarPose(R1, t1, obj_vecs, img_vecs, K)
        pose2=evaluarPose(R2, t2, obj_vecs, img_vecs, K)
        mejor_pose, num =mejor_pose_camara(pose1,pose2,len(obj_vecs))
        if num == 1: 
            k_elegido = k1 
            Z0_elegido = Z01 
            R_elegido=R1
            t_elegido=t1
        elif num == 2: 
            k_elegido = k2 
            Z0_elegido = Z02 
            R_elegido=R2
            t_elegido=t2
        else: 
            return None
        eps_nuevo=np.zeros(len(eps))
        for i in range(1,len(eps_nuevo)):
            eps_nuevo[i]=alc.producto_interno(A[i-1], k_elegido)/Z0_elegido
        diferencias=np.abs(eps_nuevo-eps)
        if max(diferencias)<atol:
            return R_elegido, t_elegido
        eps=eps_nuevo
    return R_elegido, t_elegido
        
#ejercicio 6.f:
#estimar R y t
puntos_2d = np.array([
    [525, 1000],  # Centro M0
    [403, 1095],  # Superior izquierda
    [390, 947],   # Superior derecha
    [640, 915],   # Inferior derecha
    [656, 1063]   # Inferior izquierda
])
        
MaxIt=100
atol=1e-8
R,t=aplicarPOSIT(puntos_3d, puntos_2d, matriz_K, MaxIt, atol)
print(R,t)
print("R.T @ R:")
print(R.T @ R)
print("Determinante:", np.linalg.det(R))
validos, error = evaluarPose(
    R, t, puntos_3d, puntos_2d, matriz_K)
print("Puntos válidos:", validos)
print("RMSE:", error)