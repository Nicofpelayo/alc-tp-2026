import numpy as np

### Labo 00 y otros auxiliares

def esCuadrada(A):
    return len(A) == len(A[0])

def esSimetrica(A, tol=1e-8):
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
    H_v1 = np.eye(n) - 2*(multiplicar(e1_v1, traspuesta(e1_v1)) / norma(e1_v1, 2)**2)

    if n == 2:
        S = H_v1
        D = multiplicar(H_v1, multiplicar(A, traspuesta(H_v1)))
        return S, D

    B = multiplicar(H_v1, multiplicar(A, traspuesta(H_v1)))
    AA = B[1:n,1:n]
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