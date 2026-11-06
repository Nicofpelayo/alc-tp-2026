import numpy as np

### Labo 00 y otros auxiliares

def esCuadrada(A):
    return len(A) == len(A[0])

def esSimetrica(A, tol=1e-7):
    if not esCuadrada(A):
        return False

    for f in range(0, len(A)):
        for c in range(0, len(A[0])):
            if np.abs(A[f][c] - A[c][f]) > tol:
                return False

    return True

def traspuesta(A):
    if np.ndim(A) == 1:
      #n = len(A)
      #T = np.zeros((n,1))
      #for i in range(n):
      #  T[i][0] = A[i]
      #return T
      return np.array([A])

    n, m = np.shape(A)
    T = np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T

def calcularAx(A,x,retorna_nops=False):
    n, m = np.shape(A)
    b = np.zeros(n)
    nops = 0

    for f in range(n):
        row_mult_sum = 0
        for c in range(m):
            row_mult_sum += A[f][c] * x[c]
            nops += 2
        b[f] = row_mult_sum

    if retorna_nops:
      return b, nops
    else:
      return b

def multiplicar(A,B,retorna_nops=False):
    if np.ndim(A) == 1:
       A = A.reshape((-1,1))

    B_es_vector = np.ndim(B) == 1
    if B_es_vector:
       B = B.reshape((-1,1))

    if np.shape(A)[1] != np.shape(B)[0]:
       return None
    
    n, _ = np.shape(A)
    _, q = np.shape(B)
    nops = 0

    C = np.zeros((n,q))
    for k in range(q):
      if retorna_nops:
        C[:,k], nops_cax = calcularAx(A,B[:,k], retorna_nops=True)
        nops += nops_cax
      else:
        C[:,k] = calcularAx(A,B[:,k])

    if n == 1 and  q == 1:
        C = C[0][0]
    elif B_es_vector:
      C = C[:,0]

    if retorna_nops:
      return C, nops
    else:
      return C

def triu(A):
    n, m = np.shape(A)
    U = np.copy(A)
    for i in range(m):
        U[i+1:,i] = np.zeros(n-i-1)
    return U

def tril(A):
    n, m = np.shape(A)
    L = np.copy(A)
    for i in range(n):
        L[i,i:] = np.zeros(m-i)
    L = np.eye(n) + L
    return L

### -------------------------------------------- Labo 01 --------------------------------------------

def error(x,y):
    """
    Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64
    """
    return abs(x - y)

def error_relativo(x,y):
    """
    Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64
    """
    return error(x,y) / abs(x)

def matricesIguales(A,B):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    if np.shape(A) != np.shape(B):
        return False

    rows, cols = np.shape(A)

    for i in range(0, rows):
        for j in range(0, cols):
            if error(A[i][j], B[i][j]) > 1e-15: #ad hoc
                return False

    return True

### -------------------------------------------- Labo 02 --------------------------------------------

def rota(theta):
    """
    Recibe un angulo theta y retorna una matriz de 2x2
    que rota un vector dado en un angulo theta
    """
    return np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )

def escala(s):
    """
    Recibe una tira de numeros s y retorna una matriz cuadrada de
    n x n, donde n es el tamano de s.
    La matriz escala la componente i de un vector de Rn
    en un factor s[i]
    """
    n = len(s)
    T = np.zeros((n,n))
    for i in range(0,n):
        T[i][i] = s[i]
    
    return T

def multFilaPorColumna(A,B,i,j):
    sum = 0
    for k in range(0,len(B)):
        sum += A[i][k] * B[k][j]
    return sum

def rota_y_escala(theta, s):
    """
    Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s
    """
    T_rot = rota(theta)
    T_escala = escala(s)

    T = np.zeros((2,2))
    for i in range(0,2):
        for j in range(0,2):
            T[i][j] = multFilaPorColumna(T_escala,T_rot,i,j)

    return T

def afin(theta, s, b):
    """
    Recibe un angulo theta, una tira de numeros s (en R2), y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    luego lo escala en un factor s y por ultimo lo mueve en un valor fijo b
    """
    T_rotayescala = rota_y_escala(theta, s)

    T = np.eye(3)
    T[:2, :2] = T_rotayescala # [Hasta la fila 2 exclusive, # [Hasta la col 2 exclusive]]
    T[:2,2] = b # [Hasta la fila 2 exclusive, columna 2]

    return T

def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R2), un angulo theta,
    una tira de numeros s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformacion afin a v
    """
    T = afin(theta, s, b)
    w = np.ones(3)
    w[:2] = v

    return calcularAx(T,w)[:2]

### -------------------------------------------- Labo 03 --------------------------------------------

def norma(x, p,retorna_nops=False):
  """
  Devuelve la norma p del vector x
  """
  if p == 'inf':
        return max(abs(x))

  p = np.float64(p)
  x = np.array(x)

  nops = 0
  sum = 0
  for i in range(0, len(x)):
    sum += abs(x[i])**p
    nops += 1 + p

  if retorna_nops:
    return sum**(1/p), nops
  else:
    return sum**(1/p)

def normaliza(X, p):
  """
  Recibe X, una lista de vectores no vacios, y un escalar p. Devuelve
  una lista donde cada elemento corresponde a normalizar los
  elementos de X con la norma p.
  """
  Y = []
  
  for v in X:
    norm_v = norma(v, p)
    Y.append(np.array([vi / norm_v for vi in v]))

  return Y

def normaInfDeMatriz(A):
    f,c = np.shape(A)
    max_sum = 0
    for i in range(0, c):
        sum = 0
        for j in range(0, f):
            sum += abs(A[i][j])
        if sum > max_sum:
            max_sum = sum
    return max_sum
                
def normaUnoDeMatriz(A):
    f,c = np.shape(A)
    max_sum = 0
    for j in range(0, f):
        sum = 0
        for i in range(0, c):
            sum += abs(A[i][j])
        if sum > max_sum:
            max_sum = sum
    return max_sum

def normaExacta(A, p=[1,'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2. ( c ).
    """
    if p==1:
        return normaUnoDeMatriz(A)
    if p=='inf':
        return normaInfDeMatriz(A)
    if p==[1,'inf']:
        return [normaUnoDeMatriz(A), normaInfDeMatriz(A)]
    
    return None

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q,p} y el vector x en el cual se alcanza
    el maximo.
    """
    m,n = np.shape(A)

    X = np.random.randint(-100, 100, size=(Np, n))
    #X = np.random.random((Np, n))
    X = normaliza(X,p)

    max_norm_q = 0
    for x in X:
        Ax = A@x
        norm_q = norma(Ax, q)

        if norm_q > max_norm_q:
            max_norm_q = norm_q
            max_x = x

    return (max_norm_q, max_x)

def condMC(A, p, Np):
    """
    Devuelve el numero de condicion de A usando la norma inducida p.
    """
    # Np = 10000
    A_inv = np.linalg.inv(A)
    A_p = normaMatMC(A, p, p, Np)
    A_inv_p = normaMatMC(A_inv, p, p, Np)

    return A_p[0] * A_inv_p[0]

def condExacto(A, p):
    """
    Que devuelve el numero de condicion de A a partir de la formula
    de la ecuacion (1) usando la norma p.
    """
    A_inv = np.linalg.inv(A)

    A_p = normaExacta(A, p)
    A_inv_p = normaExacta(A_inv, p)

    return A_p * A_inv_p

### -------------------------------------------- Labo 04 --------------------------------------------

def calculaLU(A):
    """
        Calcula la factorizacion LU de la matriz A y retorna las matrices L y U,
        junto con el numero de operaciones realizadas. En caso de que la matriz 
        no pueda factorizarse retorna None.
    """
    if A is None:
        return None, None, 0

    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        return None, None, cant_op
    
    for k in range(0, n-1):
        for i in range(k+1, m):
            if Ac[k][k] == 0:
                return None, None, cant_op
            
            m_ik = Ac[i][k]/Ac[k][k]
            cant_op += 1

            # Fila(i) - m_ik*Fila(k)
            f_i = Ac[i][k+1:] - m_ik*Ac[k][k+1:]
            cant_op += (n-k)*2 # It just works
            #cant_op += n*2

            Ac[i][k+1:] = f_i

            # Guardamos el coeficiente para L
            Ac[i][k] = m_ik

    L = tril(Ac)
    U = triu(Ac)
    
    return L, U, cant_op

def res_tri(L,b,inferior=True):
    """
    Resuelve el sistema Lx = b, donde L es triangular. Se puede indicar
    si es triangular inferior o superior usando el argumento
    inferior (por default asumir que es triangular inferior).
    """
    m,n = np.shape(L)
    if m != n:
        return None

    X = np.zeros(m)

    if inferior:
        for k in range(m):
            if L[k][k] == 0:
                return None

            sum = 0
            for i in range(0, k):
                sum += L[k][i]*X[i]

            X[k] = (b[k] - sum) / L[k][k]
    else:
        for k in reversed(range(m)):
            if L[k][k] == 0:
                return None

            sum = 0
            for i in range(k+1, n):
                sum += L[k][i]*X[i]

            X[k] = (b[k] - sum) / L[k][k]

    return X

def inversa(A):
    """
    Calcula la inversa de A empleando la factorizacion LU
    y las funciones que resuelven sistemas triangulares.
    """
    L, U, _ = calculaLU(A)

    if L is None or U is None:
        return None

    m, _ = np.shape(A)

    # Vemos si det(U) = 0
    if np.prod(U.diagonal()) == 0:
        return None

    L_inv = np.eye(m)
    U_inv = np.eye(m)

    for i in range(m):
        b = np.zeros(m)
        b[i] = 1
        X_L = res_tri(L,b)
        X_U = res_tri(U,b,False)

        L_inv[:, i] = X_L
        U_inv[:, i] = X_U

    return multiplicar(U_inv, L_inv)

def calculaLDV(A):
    """
    Calcula la factorizacion LDV de la matriz A, de forma tal que A = LDV,
    con L triangular inferior, D diagonal y V triangular superior. En caso 
    de que la matriz no pueda factorizarse retorna None.
    """
    L, U, _ = calculaLU(A)

    if L is None or U is None:
        return None, None, None

    U_t = traspuesta(U)

    V_t, D, _ = calculaLU(U_t)

    if V_t is None or D is None:
        return None, None, None

    return L, D, traspuesta(V_t)

def esSDP(A,atol=1e-8) :
    """
    Checkea si la matriz A es simetrica definida positiva (SDP) usando
    la factorizacion LDV.
    """
    if not esSimetrica(A,atol):
        return False

    _, D, _ = calculaLDV(A)

    for i in range(len(D)):
        if D[i][i] <= atol:
            return False

    return True

# Ej extra (Ej 5)

def calculaCholesky(A, atol=1e-10):
    """
    Verifica si la matriz es SDP y en caso afirmativo devuelve 
    la matriz R asociada a A. Caso contrario devuelve None.

    En los casos en los que A es SDP, se puede factorizar
    en la forma R·R.T, donde R es una matriz triangular inferior tal que R = L·sqrt(D),
    donde sqrt(D) es una matriz diagonal con las raices de los elementos de la diagonal
    de la matriz D, y L y D resultan de la factorización LDV de A.
    """
    if not esSDP(A):
        return None

    L, D, L_t = calculaLDV(A)

    D_sqrt = D
    for i in range(D.shape[0]):
        D_sqrt[i,i] = np.sqrt(D_sqrt[i,i])
    
    return multiplicar(L, D_sqrt) # R = L·D^{1/2}

### -------------------------------------------- Labo 05 --------------------------------------------

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    n,m = np.shape(A)
    if n != m:
      return None

    nops = 0
    Q = np.zeros((n,n))
    R = np.zeros((n,n))

    a1 = A[:,0]
    norm2_a1, nops_norma = norma(a1, 2, retorna_nops=True)
    nops += nops_norma

    Q[:,0] = a1 / norm2_a1
    nops += 1
    R[0,0] = norm2_a1

    for j in range(1, n):
      qq_j = A[:,j]
      for k in range(j):
        q_k = Q[:,k]
        R[k,j], nops_mult = multiplicar(traspuesta(q_k), qq_j, retorna_nops=True)
        nops += nops_mult
        qq_j = qq_j - R[k,j] * q_k
        nops += 2
      R[j,j], nops_norma = norma(qq_j, 2, retorna_nops=True)
      nops += nops_norma
      Q[:,j] = qq_j / R[j,j]
      nops += 1

    if retorna_nops:
      return Q, R, nops
    else:
      return Q, R

def QR_con_HH(A,tol=1e-12,extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve: 
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso


    """
    m, n = np.shape(A)
    if m < n:
        return None

    R = np.copy(A)
    Q = np.eye(m)
    if extras:
        R_matrices = [R]
        Q_matrices = [Q]
    for k in range(n):
        x = R[k:m,k]
        a = -np.sign(x[0])*norma(x,2)
        I_mk = np.eye(m-k)
        u = x - a * I_mk[:,0]
        norm2_u = norma(u,2)
        if norm2_u > tol:
            u = u / norm2_u
            H_k = I_mk - 2*multiplicar(u,traspuesta(u))
            HH_k = np.eye(m)
            HH_k[k:,k:] = H_k
            R = multiplicar(HH_k, R)
            Q = multiplicar(Q, traspuesta(HH_k))
            if extras:
                R_matrices.append(R)
                Q_matrices.append(Q)

    if extras:
        return Q, R, {"R_matrices": R_matrices, "Q_matrices": Q_matrices}
    else:
        return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol)

    return None

### -------------------------------------------- Labo 06 --------------------------------------------

def f_A(A,v):
    u = multiplicar(A,v)
    norm_u = norma(u,2)

    if norm_u > 0:
        return u/norm_u
    
    return np.zeros(len(u))

def fk_A(A,v,k):
    w = v
    for _ in range(k):
        w = f_A(A,w)
    return w

def canonico(n,i):
    v = np.zeros(n)
    v[i] = 1
    return v

def metpot2k(A, tol=1e-15, K=1000, devolver_err=False):
    """
    A una matriz de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector.
    K el numero máximo de iteraciones a realizarze.
    Retorna vector v, autovalor lambda y número de iteraciones realizadas k.
    """
    n = len(A)
    v = np.random.uniform(low=-2, high=2, size=n)
    w = fk_A(A,v,2)
    e = multiplicar(traspuesta(w), v)
    k = 0

    while abs(e-1) > tol and k < K:
        v = w
        w = fk_A(A,v,2)
        e = multiplicar(traspuesta(w), v)
        k = k + 1

    aval = multiplicar(traspuesta(w), multiplicar(A,w))
    err = e - 1

    if devolver_err:
        return v, aval, k, err
    return v, aval, k


def iszero(a, atol=1e-12):
    return np.abs(a) <= atol
def diagRH(A, tol=1e-15, K=1000):
    """
    A una matriz simétrica de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector.
    K el número máximo de iteraciones a realizarse
    retorna matriz de autovectores S y matriz de autovalores D, tal que A = S D S.T
            Si la matriz A no es simétrica, debe retornar None
    """

    if not esSimetrica(A):
        return None

    n = len(A)
    v1, lambda1, _ = metpot2k(A, tol, K)
    e1 = canonico(n,0)
    e1_v1 = e1 - v1
    norma_e1_v1 = norma(e1_v1, 2)
    if iszero(norma_e1_v1):
        H_v1 = np.eye(n)
    else:
        H_v1 = np.eye(n) - 2*(multiplicar(e1_v1, traspuesta(e1_v1)) / norma_e1_v1**2)

    if n == 2:
        S = H_v1
        D = multiplicar(H_v1, multiplicar(A, traspuesta(H_v1)))
        return S, D

    B = multiplicar(H_v1, multiplicar(A, traspuesta(H_v1)))
    AA = B[1:n,1:n]
    AA = (AA + traspuesta(AA)) / 2

    SS, DD = diagRH(AA, tol, K)
    
    D = np.eye(n)
    D[0][0] = lambda1
    D[1:, 1:] = DD

    S = np.eye(n)
    S[1:, 1:] = SS
    S = multiplicar(H_v1, S)

    return S, D

### -------------------------------------------- Labo 07 --------------------------------------------

def normalizar_columnas_suman_uno(T):
    n = len(T)
    for j in range(n):
        S_j = np.sum(T[:,j])
        if S_j == 0: # Caso borde: columna suma 0
            T[np.random.randint(n), j] = 1.0 # Elegimos una fila random de la col j para setear en 1
            S_j = 1
        T[:,j] = T[:,j] / S_j


def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0, 1]
    """
    T = np.random.uniform(low=0.0, high=1.0, size=(n,n))
    
    normalizar_columnas_suman_uno(T)
    
    return T

def transiciones_al_azar_uniformes(n, thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
        El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
        Todos los elementos de la columna j son iguales a 1 sobre el número de elementos distintos de cero en la columna.
    """
    T = np.random.uniform(low=0.0, high=1.0, size=(n,n))

    for i in range(n):
        for j in range(n):
            T[i,j] = 1.0 if T[i,j] <= thres else 0.0
    
    normalizar_columnas_suman_uno(T)
    
    return T

def nucleo(A, tol=1e-15):
    """
    A es una matriz de n x n.
    tol la tolerancia para asumir que un vector está en el núcleo
    Calcula el núcleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicación matricial),
        usando el método diagRH. El núcleo corresponde a los autovectores de autovalor con módulo <= tol.
    Retorna los autovectores en cuestión, como una matriz de n x k, con k el número de autovectores en el núcleo.
    """
    A_t = traspuesta(A)
    A_t_A = multiplicar(A_t, A)
    S, D = diagRH(A_t_A, tol)

    Nu = []

    for i in range(len(D)):
        if abs(D[i,i]) <= tol:
            Nu.append(S[:,i])

    Nu = np.array(Nu)
    return traspuesta(Nu) if len(Nu) > 0 else Nu


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: 
        lista con índices i,
        lista con índices j,
        y lista con valores A_ij de la matriz A. 
    También las dimensiones de la matriz a través de m_filas y n_columnas.
    Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir únicamente posiciones correspondientes a valores distintos de cero.
    Retorna una lista con:
    - Diccionario {(i, j): A_ij} que representa los elementos no nulos de la matriz A. 
        Los elementos con módulo menor a tol deben descartarse por default.
    - Tupla (m_filas, n_columnas) que permita conocer las dimensiones de la matriz.
    """
    if len(listado) == 0:
        return {}, (m_filas, n_columnas)
    
    indices_i, indices_j, valores_aij = listado

    res = {}
    for k in range(len(indices_i)):
        i = indices_i[k]
        j = indices_j[k]
        a_ij = valores_aij[k]

        if abs(a_ij) >= tol:
            res[(i,j)] = a_ij

    return res, (m_filas, n_columnas)

def multiplica_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crear_rala y un vector v.
    Retorna un vector w resultado de multiplicar A con v.
    """
    es_lista = isinstance(v, list)

    v = np.array(v)
    dic, shape = A
    n, m = shape
    w = np.zeros(n)

    for (i,j), a_ij in dic.items():
        w[i] += a_ij * v[j]

    # Naive
    # for i in range(n):
    #     sum = 0.0
    #     for j in range(m):
    #         if (i,j) in dic:
    #             sum += dic[(i,j)] * v[j]
    #     w[i] = sum

    return w.tolist() if es_lista else w

### -------------------------------------------- Labo 08 --------------------------------------------

def svd_reducida(A,k="max",tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """
    m, n = np.shape(A)
    A_t = traspuesta(A)
    U_es_V = False

    if m < n:
        A_aux = A
        A = A_t
        A_t = A_aux
        U_es_V = True
    
    AtA = multiplicar(A_t,A)

    S, D = diagRH(AtA, tol=tol)

    r = 0
    hatV = []
    hatSig = []
    while r < len(D) and D[r,r] >= tol:
        hatV.append(np.asarray(S[:,r]))
        hatSig.append(np.sqrt(D[r,r]))  # σ^2 = (D_ii)^2 = (Σ_ii)^2 
        r += 1

    hatV = traspuesta(np.array(hatV))
    hatSig = np.array(hatSig)  #  Recordar: hatSig es un vector en svd reducida

    hatU = multiplicar(A,hatV)

    for i in range(len(hatSig)):
        hatU[:,i] = hatU[:,i] / hatSig[i]

    if U_es_V:
        temp = hatV
        hatV = hatU
        hatU = temp

    #return hatU, hatSig, hatV

    # k > r, ver Ej3
    # En caso de k sea mayor a r (es decir, que durante el procedimiento se encuentre
    # un valor singular nulo), el procedimiento debe detenerse en ese punto,
    # retornando la factorización reducida como en el Ejercicio 1.
    if k == "max" or k > r: 
        return hatU, hatSig, hatV

    hatU = hatU[:,:k]
    hatSig = hatSig[:k]
    hatV = hatV[:,:k]

    return hatU, hatSig, hatV


### -------------------------------------- Aux extras para el TP --------------------------------------

def minimo_cuadrado_homogeneo(A):
    """
    Resuelve min_{||b||_2 = 1} Ab
    """
    # Con SVD sale igual, pero hay que tomar la svd COMPLETA,
    # NO la svd_reducida. La reducida descarta los que tengan σ < tol,
    # y como el b queremos que esté en el nucleo, tendra σ ≈ 0 y es descartado
    #
    # En su lugar, planteamos la diagonalización de Q.T Q usando diagRH,
    # que tambien deja ordenados los autovectores en orden descendente
    # en relacion a sus |λ| = σ², y seguimos la misma idea de tomar el 
    # de más a la derecha
    
    S, _ = diagRH(multiplicar(traspuesta(A),A))
    b = S[:, -1]
    return b

def pseudoinversa(A):
    """
    Devuelve la pseudoinversa de A
    """
    hatU, hatSig, hatV = svd_reducida(A)

    # hatSig cuadrada y diagonal por svd_reducida
    r = len(hatSig)
    Sig_inv = np.zeros((r, r))
    for i in range(r):
        Sig_inv[i,i] = 1 / hatSig[i]

    return multiplicar(multiplicar(hatV, Sig_inv), traspuesta(hatU))

def producto_vectorial(u,v):
    """
    u y v dos vectores en R^3
    Devuelve w = u x v
    """
    w1 = u[1] * v[2] - u[2] * v[1]
    w2 = u[2] * v[0] - u[0] * v[2]
    w3 = u[0] * v[1] - u[1] * v[0]

    return np.array([w1, w2, w3])

def producto_interno(u,v):
    """
    u y v dos vectores en R^n
    Devuelve en número real resultado del producto interno u.T v
    """
    return multiplicar(traspuesta(u), v)



### -------------------------------------------- TP: Ejercicio 2 --------------------------------------------

def obtenerQ3d(imgPts, objPts):
    """
    imgPts los puntos (u, v) en la imagen, como matriz de n x 2 o como vector [u1, v1, u2, v2, ...]
    objPts los mismos n puntos (U, V, W) en el mundo, como matriz de n x 3
    Retorna la matriz Q de 2n x 12 del sistema Qm = 0 (ecuación 6), con dos filas por cada punto
    """
    imgPts = np.asarray(imgPts).reshape(-1, 2)
    n = objPts.shape[0]
    Q = np.zeros((2 * n, 12))
    for i in range(n):
        # Para cada punto generamos sus 2 filas:
        #   (P_i^T, 0^T, -u_i P_i^T) y (0^T, P_i^T, -v_i P_i^T), con P_i = (U, V, W, 1)
        P_i = np.hstack((objPts[i], 1))
        u_i = imgPts[i, 0]
        v_i = imgPts[i, 1]

        Q[i * 2]     = np.hstack((P_i, [0, 0, 0, 0], -1 * u_i * P_i))
        Q[i * 2 + 1] = np.hstack(([0, 0, 0, 0], P_i, -1 * v_i * P_i))
    return Q

def resolverDLT3D(Q):
    """
    Q la matriz de 2n x 12 armada con obtenerQ3d
    Retorna la matriz de proyección M de 3 x 4, resolviendo min_{||m||_2 = 1} ||Qm||_2 (Ejercicio 1 b).
    M queda determinada a menos de un factor de escala
    """
    # m es el vector singular derecho asociado al menor valor singular de Q
    m = minimo_cuadrado_homogeneo(Q)
    # m = (m1, m2, m3) con m_j^T la fila j de M
    return m.reshape(3, 4)

def signo(x):
    """
    Recibe un numero x y retorna su signo: -1 si x < 0, 1 si x > 0 y 0 si x = 0
    """
    if x < 0:
        return -1
    elif x > 0:
        return 1
    return 0

def devolverMatricesCalibracion3d(M, restringir_K12=False, restringir_K11_K22=False):
    """
    M la matriz de proyección de 3 x 4 (M = K [R | t])
    restringir_K12 si es True, impone K12 = 0 (sin skew)
    restringir_K11_K22 si es True, impone alpha_u = alpha_v, reemplazando K11 y K22 por su promedio
    Retorna la matriz de rotación R de 3 x 3 (ortonormal y con det(R) = 1), el vector de traslación t
    y la matriz de intrínsecos K de 3 x 3 (triangular superior y normalizada para que K33 = 1)
    """
    # M = [KR | Kt]: el bloque 3x3 de la izquierda es KR y la cuarta columna es Kt
    matriz_KR = M[:, :3]
    ultima_columna_M = M[:, 3]

    # (KR)^-1 = R^-1 K^-1 = R^T K^-1, con R^T ortogonal y K^-1 triangular superior,
    # así que una factorización QR de (KR)^-1 nos da Q = R^T y la "R" del QR = K^-1
    R_traspuesta, K_inv = QR_con_GS(inversa(matriz_KR))
    R = traspuesta(R_traspuesta)
    K = inversa(K_inv)

    # Kt = cuarta columna de M  =>  t = K^-1 (cuarta columna de M)
    t = calcularAx(K_inv, ultima_columna_M)

    # La factorización QR es única salvo signos: KR = (K D)(D R) para cualquier
    # D = diag(±1, ±1, ±1). Elegimos D para que K11/K33 > 0, K22/K33 > 0 y det(R) = 1
    s1 = signo(K[0, 0] * K[2, 2])
    s2 = signo(K[1, 1] * K[2, 2])
    sR = signo(determinante_3x3(R))
    d3 = s1 * s2 * sR
    d1 = s1 * d3
    d2 = s2 * d3
    D = escala([d1, d2, d3])

    R = multiplicar(D, R)
    K = multiplicar(K, D)
    t = calcularAx(D, t)

    # M se conoce salvo escala: normalizamos para que K33 = 1
    K = K / K[2, 2]

    # Ejercicio 2.d: restricciones opcionales sobre K
    if restringir_K11_K22:
        promedio = (K[0, 0] + K[1, 1]) / 2
        K[0, 0] = promedio
        K[1, 1] = promedio
    if restringir_K12:
        K[0, 1] = 0

    return R, t, K

### -------------------------------------------- TP: Ejercicio 3 --------------------------------------------

def obtenerQ2d(imgPts, objPts):
    """
    imgPts los puntos (u, v) en la imagen, como matriz de n x 2
    objPts los mismos n puntos (U, V) en el plano, como matriz de n x 2
    Retorna la matriz Q de 2n x 9 del sistema Qh = 0 (ecuación 11), con dos filas por cada punto
    """
    n = objPts.shape[0]
    Q = np.zeros((2 * n, 9))
    for i in range(n):
        # Para cada punto generamos sus 2 filas
        P_i = np.hstack((objPts[i], 1))
        u_i = imgPts[i, 0]
        v_i = imgPts[i, 1]

        Q[i * 2] = np.hstack((P_i, [0, 0, 0], -1 * u_i * P_i))
        Q[i * 2 + 1] = np.hstack(([0, 0, 0], P_i, -1 * v_i * P_i))
    return Q

def obtenerTHartley(pts):
    """
    pts los puntos a normalizar, como matriz de n x 2
    Calcula la normalización de Hartley: traslada los puntos para que su centroide quede en el origen
    y los escala para que la distancia promedio al origen sea sqrt(2)
    Retorna la matriz T de 3 x 3 de la normalización (en coordenadas homogéneas) y su inversa T_inv
    """
    # Calculo de centroide (traslacion)
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = np.mean(xs)
    cy = np.mean(ys)
    c = np.array([cx, cy])

    # Promedio de distancias (escala)
    distancias = [norma(pts[i] - c, 2) for i in range(len(pts))]
    dist_promedio = np.mean(distancias)
    s = np.sqrt(2) / dist_promedio

    #      [1 0 −c_x]         [s 0 0]
    # Tr = [0 1 −c_y],   S =  [0 s 0]
    #      [0 0   1 ]         [0 0 1]
    #
    #            [s 0 -s·c_x]
    # T = S·Tr = [0 s -s·c_y]
    #            [0 0    1  ]
    #
    #                              [1/s  0   c_x]
    # T^{-1} = Tr^{-1} · S^{-1}  = [ 0  1/s  c_y]
    #                              [ 0   0    1 ]
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
    """
    imgPts los puntos de la imagen ya normalizados, como matriz de n x 2
    objPts los puntos del plano ya normalizados, como matriz de n x 2
    Retorna la homografía de 3 x 3 entre los puntos normalizados, resolviendo min_{||h||_2 = 1} ||Qh||_2
    """
    # Q de tamaño 2Nx9
    Q = obtenerQ2d(imgPts, objPts)
    h = minimo_cuadrado_homogeneo(Q)

    return h.reshape(3, 3)

def normalizarPts(T, pts):
    """
    T una matriz de 3 x 3 (por ejemplo, la de obtenerTHartley)
    pts los puntos a transformar, como matriz de n x 2
    Retorna la matriz de n x 2 con los puntos transformados por T, usando coordenadas homogéneas
    """
    normalizados = []

    for i in range(pts.shape[0]):
        p_i = np.hstack((pts[i], 1))
        normalizados.append(multiplicar(T, p_i))

    # La tercera coordenada queda en 1 porque T es afín
    return np.array(normalizados)[:, :-1]

def resolverDLT2D(imgPts, objPts):
    """
    imgPts los puntos (u, v) en la imagen, como matriz de n x 2 (n >= 4, sin 3 alineados)
    objPts los mismos n puntos (U, V) en el plano, como matriz de n x 2
    Retorna la homografía H de 3 x 3 tal que [u, v, 1] ~ H [U, V, 1] (a menos de un factor de escala),
    calculada con DLT 2D y normalización de Hartley
    """
    # Normalizar usando Hartley
    Timg, Timg_inv = obtenerTHartley(imgPts)
    Tworld, _ = obtenerTHartley(objPts)

    imgPtsNormalizados = normalizarPts(Timg, imgPts)
    objPtsNormalizados = normalizarPts(Tworld, objPts)

    H_hat = resolverDLT2D_PtsNormalizados(imgPtsNormalizados, objPtsNormalizados)

    # Deshacemos la normalización: H = Timg^-1 H_hat Tworld
    return multiplicar(Timg_inv, multiplicar(H_hat, Tworld))

def v_ij(H, i, j):
    """
    H una homografía de 3 x 3
    i, j los índices de las columnas de H (0, 1 o 2)
    Retorna el vector v_ij de 6 elementos tal que c_i^T B c_j = v_ij b, con c_i y c_j las columnas i y j de H
    y b = (B11, B12, B22, B13, B23, B33)
    """
    # B simétrica (B_ij = B_ji), entonces
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
    Hlst una lista con las N homografías de 3 x 3 (una por vista, obtenidas con resolverDLT2D)
    Retorna la matriz A de 2N x 6 del algoritmo de Zhang, con las filas v12 y v11 - v22 de cada vista
    """
    N = len(Hlst)
    A = np.zeros((2 * N, 6))

    for k in range(N):
        H = Hlst[k]

        # Calculamos los v_ij requeridos por el paso e) del algoritmo de Zhang
        v12 = v_ij(H, 0, 1)
        v11 = v_ij(H, 0, 0)
        v22 = v_ij(H, 1, 1)

        # Asignamos las dos filas correspondientes en la matriz A
        A[2 * k]     = v12
        A[2 * k + 1] = v11 - v22

    return A

def calcularMatrizK2D(A, restringir_K12=False, restringir_K11_K22=False):
    """
    A la matriz de 2N x 6 armada con armarA
    restringir_K12 si es True, impone K12 = 0 (sin skew)
    restringir_K11_K22 si es True, impone alpha_u = alpha_v, reemplazando K11 y K22 por su promedio
    Retorna la matriz de intrínsecos K de 3 x 3 (triangular superior y normalizada para que K33 = 1)
    """
    # b: solución de Ab = 0 con ||b|| = 1
    b = minimo_cuadrado_homogeneo(A)

    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    # B se conoce salvo signo: si no es definida positiva, la damos vuelta
    if not esSDP(B):
        B = (-1) * B

    # B = K^-T K^-1 = L L^T  =>  K = (L^T)^-1
    L = calculaCholesky(B)

    K = inversa(traspuesta(L))

    # Como B se conoce solo salvo escala, se normaliza K dividiendola
    # por K_33 para que K_33 = 1
    K = K / K[2, 2]

    if restringir_K11_K22: 
        promedio = (K[0,0] + K[1,1]) / 2 
        K[0,0] = promedio  
        K[1,1] = promedio 
    if restringir_K12: 
        K[0,1] = 0

    return K

def obtenerPose2D(Hj, K):
    """
    Hj la homografía de 3 x 3 de la vista j
    K la matriz de intrínsecos de 3 x 3
    Retorna la matriz de rotación Rj de 3 x 3 y el vector de traslación tj de la vista j
    """
    K_inv = inversa(K)
    c1, c2, c3 = Hj[:, 0], Hj[:, 1], Hj[:, 2]
    K_inv_c1 = multiplicar(K_inv, c1)
    K_inv_c2 = multiplicar(K_inv, c2)
    K_inv_c3 = multiplicar(K_inv, c3)

    s = norma(K_inv_c1, 2)
    lambd = 1 / s

    r1 = lambd * K_inv_c1
    r2 = lambd * K_inv_c2
    r3 = producto_vectorial(r1,r2)
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
    Rj, _ = QR_con_GS(Rj)

    return Rj, tj

def aplicarHomografia(H, pts):
    """
    H una homografía de 3 x 3
    pts los puntos 2D a transformar, como matriz de n x 2
    Retorna la matriz de n x 2 con los puntos transformados: para cada punto (x, y) calcula
    λ[u, v, 1] = H [x, y, 1]^T y divide por la tercera coordenada (λ)
    """
    transformados = np.zeros((pts.shape[0], 2))
    for i in range(pts.shape[0]):
        # Coordenadas homogéneas del punto
        p_homogeneo = np.array([pts[i, 0], pts[i, 1], 1.0])
        q = multiplicar(H, p_homogeneo)
        # Deshomogeneizamos dividiendo por la tercera coordenada
        transformados[i, 0] = q[0] / q[2]
        transformados[i, 1] = q[1] / q[2]
    return transformados

### -------------------------------------------- TP: Ejercicio 5--------------------------------------------

def obtenerHsK_coplanar(imgPtsList, objPts, restringir_K12=False, restringir_K11_K22=False):
    """
    imgPtsList una lista con los puntos (u, v) en la imagen de cada vista, cada uno como matriz de n x 2
    objPts los mismos n puntos (U, V) en el plano del mundo, como matriz de n x 2
    restringir_K12, restringir_K11_K22 las restricciones opcionales sobre K (ver calcularMatrizK2D)
    Retorna la lista con la homografía de cada vista y la matriz de intrínsecos K de 3 x 3
    """
    Hs = []
    for imgPts in imgPtsList:
        H = resolverDLT2D(imgPts, objPts)
        Hs.append(H)

    A = armarA(Hs)
    K = calcularMatrizK2D(A, restringir_K12, restringir_K11_K22)
    return Hs, K

def proyectarPuntosUVW(ptsObjList, K, R, t):
    """
    ptsObjList los puntos (U, V, W) del mundo, como matriz de n x 3
    K la matriz de intrínsecos de 3 x 3
    R, t la pose de la cámara (matriz de rotación de 3 x 3 y vector de traslación)
    Retorna la matriz de n x 2 con las proyecciones (u, v) de los puntos en la imagen (ecuación 3)
    """
    Rt = np.hstack((R, t.reshape(-1,1)))
    ptsProyectados = []
    for ptsUVW in ptsObjList:
        ptUVW1 = np.hstack((ptsUVW, 1))
        ptXYZ = multiplicar(Rt, ptUVW1)
        pt_uvlambd = multiplicar(K, ptXYZ)
        pt_uv = pt_uvlambd / pt_uvlambd[2]
        ptsProyectados.append(pt_uv[:-1])

    return np.array(ptsProyectados)



### -------------------------------------------- TP: Ejercicio 6 --------------------------------------------

#ejercicio 6.b:
    
def raicesCuadradasDeCCuadrado(Re, Im):
    """
    Re la parte real de C^2
    Im la parte imaginaria de C^2
    Retorna lambd y mu, números reales tales que C = lambd + i*mu es una raíz cuadrada de C^2.
    La otra raíz es (-lambd, -mu), que se usa en aplicarPOSIT
    """
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
    if iszero(r):
        return 0, 0
    
    if Re > 0:
        lambd = np.sqrt((r + Re) / 2)
        mu = Im / (2*lambd)
    else:
        mu = np.sqrt((r - Re) / 2)
        lambd = Im / (2*mu)

    return lambd, mu


def calculoLambdaMu(I0,J0):
    """
    I0 la solución particular de AI = b_u, obtenida con la pseudoinversa de A
    J0 la solución particular de AJ = b_v, obtenida con la pseudoinversa de A
    Retorna lambd y mu, los coeficientes para construir I = I0 + lambd*n y J = J0 + mu*n
    de forma que I y J sean ortogonales y de igual norma (caso coplanar de POSIT).
    La otra solución, (-lambd, -mu), se obtiene cambiando ambos signos en aplicarPOSIT
    """
 
    # C = λ + iμ
    # C^2 = λ^2 − µ^2 + 2i λµ
    # C^2 = (λ^2 − µ^2) + i (2λµ)
    # C^2 = (||J0||^2 − ||I0||^2) + i (2 (-I0.T J0))
    # C^2 = (||J0||^2 − ||I0||^2) + i (-2 I0.T J0)

    # Calculo parte real
    norm2_I0 = norma(I0, 2)
    norm2_J0 = norma(J0, 2)
    norm2_I0_cuad = norm2_I0 * norm2_I0
    norm2_J0_cuad = norm2_J0 * norm2_J0
    Re = norm2_J0_cuad - norm2_I0_cuad

    # Calculo imaginaria
    Im = (-2) * producto_interno(I0, J0)

    lambd, mu = raicesCuadradasDeCCuadrado(Re, Im)
    return lambd, mu

#ejercicio 6.c:
    
def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    """
    Is el vector I = I0 + lambd*n
    Js el vector J = J0 + mu*n
    fx la distancia focal de la cámara
    m0 las coordenadas 2D centradas de la proyección de M0
    M0 las coordenadas 3D del punto de referencia
    Retorna una posible pose de la cámara: los vectores unitarios i, j, k (filas de R), la profundidad Z0
    del punto de referencia, la matriz de rotación R (ortogonal y con det(R) = 1) y el vector de traslación t
    """
    norm_I = norma(Is, 2)
    norm_J = norma(Js, 2)
    #normalizamos los vectores para que sean unitarios
    i = Is / norm_I
    j = Js / norm_J
    k = producto_vectorial(i,j)
    Z0 = fx / norm_I

    R = np.array([i,j,k])

    # Obtenemos t utilizando la ecuación 1 del TP
    X0 = (Z0 / fx) * m0[0] # x0
    Y0 = (Z0 / fx) * m0[1] # y0
    T = np.array([X0, Y0, Z0])
    
    t=T-calcularAx(R, M0)
    
    return i,j,k,Z0,R,t


#ejercicio 6.d:

def RMSE(m1,m2):
    """
    m1 las coordenadas 2D predichas con una pose candidata (R, t)
    m2 las coordenadas 2D observadas en la imagen
    Retorna la raíz del error cuadrático medio entre ambos conjuntos de puntos (error de reproyección, en píxeles).
    Cuanto más chico, mejor es la pose
    """
    sumatoria=0
    for i in range(len(m1)):
        norma_vector=norma(m1[i]-m2[i],2)
        sumatoria+=norma_vector**2 
    valor=np.sqrt(sumatoria/len(m1))
    return valor
                  
        
def evaluarPose(R, t, obj_vecs, img_vecs, K):
    """
    R la matriz de rotación de la cámara
    t el vector de traslación
    obj_vecs las coordenadas de los puntos 3D
    img_vecs las coordenadas 2D observadas en la imagen
    K la matriz intrínseca de la cámara
    Retorna la cantidad de proyecciones válidas (puntos 3D delante de la cámara, Z > 0) y el RMSE
    respecto de img_vecs, calculado sobre las proyecciones válidas
    """
    proyecciones_validas=0 
    proyecciones=[]
    comparaciones_validas=[]
    for i in range(len(obj_vecs)): #para cada punto 3D (del mundo real)
        #calculamos las coordenadas del punto 3D respecto al sistema de referencia de la cámara
        punto=obj_vecs[i]
        coords_camara=calcularAx(R,punto) + t
        #nos fijamos si el punto está delante de la cámara (Z > 0)
        if coords_camara[2]>0:
            proyecciones_validas+=1 
            #calculamos las coords homogéneas de la proyección 2D:(lambda*u, lambda*v, lambda)
            proyeccion=calcularAx(K,coords_camara) 
            #según la ecuación 2 del TP, lambda = Zi.
            #dividimos por el factor de escala para obtener (u, v) en píxeles
            proyeccion[0]=proyeccion[0]/proyeccion[2] 
            proyeccion[1]=proyeccion[1]/proyeccion[2]
            proyecciones.append(proyeccion[:2]) #guardamos los (u,v)
            #seleccionamos los puntos 2D que podemos comparar
            comparaciones_validas.append(img_vecs[i]) 
    if proyecciones_validas==0:
        return 0, np.inf #si no hay proyecciones validas, asignamos error infinito
    return proyecciones_validas, RMSE(proyecciones,comparaciones_validas)

#ejercicio 6.e:

def construir_A(obj_vecs):
    """
    obj_vecs las coordenadas 3D de los puntos del objeto, con M0 como primer punto
    Retorna la matriz A cuyas filas son los vectores Mi - M0, para todos los puntos distintos de M0
    """
    M0=obj_vecs[0] #punto de referencia
    matriz_A=[]
    for i in range(1,len(obj_vecs)):
        vector=obj_vecs[i]-M0 
        matriz_A.append(vector)
    return np.array(matriz_A)
        
def centrar_coords(img_vecs,K):
    """
    img_vecs las coordenadas 2D (u, v) de los puntos de la imagen
    K la matriz intrínseca de la cámara
    Retorna las coordenadas (x, y) centradas respecto del punto principal (u0, v0) de la cámara:
    x_i = u_i - u0, y_i = v_i - v0
    """
    #buscamos el punto principal (u0, v0) en la matriz K
    u0 = K[0][2]
    v0 = K[1][2]
    coords_centradas = []
    for i in range(len(img_vecs)):
        #centramos las coords
        x = img_vecs[i][0] - u0
        y = img_vecs[i][1] - v0
        coords_centradas.append(np.array([x,y]))
    return coords_centradas

def calcular_b(coords,e,parametro="u"):
    """
    coords las coordenadas 2D centradas de los puntos de la imagen
    e el vector con los factores de corrección de perspectiva epsilon_i actuales
    parametro indica si se construye b_u ("u") o b_v ("v")
    Retorna el vector de términos independientes b_u o b_v de los sistemas AI = b_u y AJ = b_v de POSIT
    """
    b=[]
    #empezamos en 1 porque coords[0] corresponde al punto de referencia M0
    for i in range(1,len(coords)): 
        if parametro=="u": 
            b_i=coords[i][0]*(1+e[i])-coords[0][0]
        else:
            b_i=coords[i][1]*(1+e[i])-coords[0][1]
        b.append(b_i)
    return np.array(b)

def mejor_pose_camara(pose1,pose2,n):
    """
    pose1 la cantidad de proyecciones válidas y el RMSE de la primera pose
    pose2 la cantidad de proyecciones válidas y el RMSE de la segunda pose
    n la cantidad total de puntos 3D
    Descarta las poses que no tienen todas las proyecciones válidas y, si quedan las dos, elige la de menor RMSE.
    Retorna la evaluación de la pose elegida y su número (1 o 2), para recuperar sus parámetros en aplicarPOSIT.
    Si ninguna pose tiene todas las proyecciones válidas, retorna (None, None)
    """
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
    """
    obj_vecs las coordenadas 3D de los puntos del objeto
    img_vecs las coordenadas 2D observadas en la imagen
    K la matriz intrínseca de la cámara
    MaxIt la cantidad máxima de iteraciones
    atol la tolerancia para determinar la convergencia
    Aplica los pasos de la sección "Iteración POSIT" del TP para estimar la pose de la cámara
    respecto del sistema de coordenadas del objeto.
    Retorna la matriz de rotación R y el vector de traslación t estimados. Si no encuentra una pose válida,
    retorna None. Si alcanza MaxIt sin converger, retorna la última pose estimada
    """
    #inicializamos el algoritmo asumiendo una proyección ortográfica escalada
    eps = np.zeros(len(obj_vecs))
    #construimos A para resolver los sistemas(AI=b_u, AJ=b_v) con su pseudo-inversa
    A = construir_A(obj_vecs)
    #centramos las coordenadas 2D observadas en la imagen
    coords = centrar_coords(img_vecs, K)
    pseudo_inv_A = pseudoinversa(A)
    #obtenemos un vector unitario n del núcleo de A (An=0)
    #para construir I=I0+lambda*n y J=J0+mu*n
    Nu = nucleo(A)
    n = Nu[:, 0]
    n = n / norma(n, 2)
    #obtenemos parámetros para aplicar calcularSalidaPOSIT()
    fx = K[0, 0]
    m0 = coords[0]
    M0 = obj_vecs[0]
    for j in range(MaxIt):
        #calculamos b_u, b_v para resolver el sistema lineal con la pseudo-inversa
        b_u = calcular_b(coords, eps, "u")
        b_v = calcular_b(coords, eps, "v")
        I0 = calcularAx(pseudo_inv_A, b_u)
        J0 = calcularAx(pseudo_inv_A, b_v)
        #sol1:
        lambda1,mu1 = calculoLambdaMu(I0, J0)
        #sol2: cambiamos los signos, para obtener la segunda raíz compleja
        lambda2,mu2 = -lambda1,-mu1
        #calculamos I=I0+lambda*n, J=J0+mu*n para cada lambda, mu
        Is = [I0+lambda1*n,I0+lambda2*n]
        Js = [J0+mu1*n,J0+mu2*n]
        #calculamos las poses a partir de las dos soluciones
        i1, j1, k1, Z01, R1, t1 = calcularSalidaPOSIT(Is[0], Js[0], fx, m0, M0)
        i2, j2, k2, Z02, R2, t2 = calcularSalidaPOSIT(Is[1], Js[1], fx, m0, M0)
        #evaluamos las poses:
        pose1=evaluarPose(R1, t1, obj_vecs, img_vecs, K)
        pose2=evaluarPose(R2, t2, obj_vecs, img_vecs, K)
        #nos quedamos con la mejor pose
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
        #actualizamos los factores de corrección de perspectiva epsilon_i
        for i in range(1,len(eps_nuevo)): 
            eps_nuevo[i]=producto_interno(A[i-1], k_elegido)/Z0_elegido
        #verificamos si las correcciones convergieron
        diferencias=np.abs(eps_nuevo-eps) 
        if max(diferencias)<atol:
            return R_elegido, t_elegido
        eps=eps_nuevo #si no converge actualizamos eps, y sigue iterando
    return R_elegido, t_elegido
        
def determinante_3x3(A):
    """
    A una matriz de 3 x 3
    Retorna el determinante de A, desarrollado por la primera fila
    """
    # Desarrollamos el determinante por la primera fila
    valor = (A[0,0] * (A[1,1]*A[2,2] - A[1,2]*A[2,1])
           - A[0,1] * (A[1,0]*A[2,2] - A[1,2]*A[2,0])
           + A[0,2] * (A[1,0]*A[2,1] - A[1,1]*A[2,0]))

    return valor

