import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import alc
from ejercicio3 import resolverDLT2D

filepath_base = '../template-alumnos/futbol'


def aplicarHomografia(H, pts):
    """
    Aplica la homografía H a la lista pts 2D

    Parámetros:
        H: matriz 3x3 homografía
        pts: array (n, 2) puntos 2D a transformar
    Devuelve:
        array (n, 2) con los puntos transformados. Para cada punto p = (x, y)
        se calcula q = H [x, y, 1]^T y se divide por la tercera coordenada (pasaje de coordenadas homogéneas a cartesianas).
    """
    transformados = np.zeros((pts.shape[0], 2))
    for i in range(pts.shape[0]):
        # Coordenadas homogéneas del punto
        p_homogeneo = np.array([pts[i, 0], pts[i, 1], 1.0])
        q = alc.multiplicar(H, p_homogeneo)
        # Deshomogeneizamos dividiendo por la tercera coordenada
        transformados[i, 0] = q[0] / q[2]
        transformados[i, 1] = q[1] / q[2]
    return transformados


def circuloEnCancha(centro, radio, n):
    """
    Puntos de una circunferencia sobre el piso, en coordenadas de la cancha

    Parámetros:
        centro: (U0, V0) centro de la circunferencia (m)
        radio: radio (m)
        n: cantidad de divisiones de cada semicircunferencia
    Devuelve:
        array (2n+2, 2) con los puntos (U, V), recorriendo la circunferencia en orden
        (primero la mitad de arriba y después la de abajo) para poder dibujarla con plt.plot.
        Sin usar seno ni coseno: de (U - U0)^2 + (V - V0)^2 = r^2 despejamos V = V0 ± sqrt(r^2 - (U - U0)^2)
    """
    U0, V0 = centro
    # Valores de U equiespaciados entre U0 - r y U0 + r
    U = U0 - radio + 2 * radio * np.arange(n + 1) / n
    # Altura de la semicircunferencia (np.abs evita negativos chicos por redondeo en los extremos)
    altura = np.sqrt(np.abs(radio**2 - (U - U0)**2))
    # Mitad de arriba (U creciente) y mitad de abajo (U decreciente) para cerrar la curva
    mitad_arriba = np.vstack([U, V0 + altura]).T
    mitad_abajo = np.vstack([U[::-1], V0 - altura[::-1]]).T
    return np.vstack([mitad_arriba, mitad_abajo])


# ---------------------------------------------------------------------------
# Ejercicio 4.a: distancia de Enzo a la línea del arco
# ---------------------------------------------------------------------------

# Coordenadas del plano estadio_azteca.png: origen en la esquina inferior izq, U a lo largo (0..105 m) y
# V a lo ancho (0..68 m).Enzo patea hacia el arco de U = 0; la cámara mira hacia ese arco, así que lo
# que en la foto está a la izquierda tiene V menor

# Puntos de referencia sobre el piso (datos del plano):
#   arco de 7.32 m              -> palos en V = 34 ± 3.66
#   área chica de 5.5 x 18.32 m -> V = 34 ± 9.16
#   punto penal a 11m           -> (11, 34)
#   medialuna de radio 9.15 m centrada en el penal, corta la línea del área
#   (U = 16.5) en V = 34 ± sqrt(9.15^2 - 5.5^2) = 34 ± 7.31
objPts = np.array([
    [0.0,  30.34],   # 0: palo izquierdo
    [0.0,  37.66],   # 1: palo derecho
    [5.5,  24.84],   # 2: área chica, esquina frontal izq
    [5.5,  43.16],   # 3: área chica, esquina frontal der
    [11.0, 34.0],    # 4: punto penal
    [16.5, 26.69],   # 5: medialuna ∩ línea del área, izq
    [16.5, 41.31],   # 6: medialuna ∩ línea del área, der
])

# Los mismos puntos en la foto (píxeles)
imgPts = np.array([
    [388.0, 176.0],
    [652.0, 184.5],
    [157.0, 231.0],
    [867.0, 250.0],
    [511.0, 317.0],
    [146.0, 406.0],
    [851.0, 422.0],
])

# Pie de apoyo de Enzo al patear (píxeles)
pie_enzo_img = np.array([[810.0, 545.0]])



# ---------------------------------------------------------------------------
# Ejercicio 4.b: círculos de 1m de rodeando a los jugadores argentinos
# ---------------------------------------------------------------------------

# Jugadores argentinos en la foto (píxeles)
pies_argentinos_img = np.array([
    [107.0, 540.0],   # Julian, abajo a la izquierda
    [244.0, 404.0],   # Lautaro, sobre la línea del área
    [388.0, 297.0],   # Cristian Romero, el de atrás (pies tapados, estimado)
    [375.0, 328.0],   # Nico González, el de adelante
    [853.0, 352.0],   # Alexis
    [810.0, 545.0],   # Enzo
])

if __name__ == '__main__':
    img = mpimg.imread(f'{filepath_base}/test-gol-enzo.png')

    # Paso 1: verificar que los puntos estén bien marcados y en orden
    plt.figure(figsize=(14, 9))
    plt.imshow(img)
    plt.plot(imgPts[:, 0], imgPts[:, 1], 'ro', markersize=4)
    for i in range(imgPts.shape[0]):
        plt.text(imgPts[i, 0] + 6, imgPts[i, 1] - 6, str(i), color='yellow', fontsize=11, weight='bold')
    plt.title('Puntos de referencia')
    plt.show()

    # Paso 2: homografía cancha -> imagen y su inversa
    H = resolverDLT2D(imgPts, objPts)
    H_inv = alc.inversa(H)

    # Paso 3: validación. Proyectamos el rectángulo del área chica: sus dos esquinas sobre la línea de meta
    # no se usaron para estimar H, así que si caen sobre las esquinas reales de la foto, H es confiable.
    area_chica = aplicarHomografia(H, np.array([[0.0, 24.84], [5.5, 24.84], [5.5, 43.16], [0.0, 43.16]]))
    plt.figure(figsize=(14, 9))
    plt.imshow(img)
    plt.plot(area_chica[:, 0], area_chica[:, 1], 'c-', linewidth=1.5)
    plt.title('Validación: área chica proyectada con H')
    plt.show()

    # Paso 4: llevamos el pie de Enzo a la cancha con H^-1
    U_enzo, V_enzo = aplicarHomografia(H_inv, pie_enzo_img)[0]
    print(f'Enzo en la cancha: U = {U_enzo:.2f} m, V = {V_enzo:.2f} m')

    # Paso 5: distancia perpendicular a la línea de meta (U = 0)
    distancia = np.abs(U_enzo)
    print(f'Distancia a la línea del arco: {distancia:.2f} m')

    # 4.b: cada pie a la cancha con H^-1, círculo de 1m en la cancha y proyección a la foto con H
    pies_cancha = aplicarHomografia(H_inv, pies_argentinos_img)
    plt.figure(figsize=(14, 9))
    plt.imshow(img)
    for i in range(pies_cancha.shape[0]):
        circulo_img = aplicarHomografia(H, circuloEnCancha(pies_cancha[i], 1.0, 50))
        plt.plot(circulo_img[:, 0], circulo_img[:, 1], 'y-', linewidth=1.5)
        print(f'Jugador {i}: U = {pies_cancha[i, 0]:.2f} m, V = {pies_cancha[i, 1]:.2f} m')
    plt.title('Jugadores argentinos con círculos de 1m de radio')
    plt.show()
