import numpy as np
import alc

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

    R = np.array(i,j,k)

    # Obtenemos t
    X0 = (Z0 / fx) * m0[0] # x0
    Y0 = (Z0 / fx) * m0[1] # y0
    T = np.array(X0, Y0, Z0)
    

