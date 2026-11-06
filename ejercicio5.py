# -*- coding: utf-8 -*-

import numpy as np
import ejercicio3 as ej3
import pandas as pd
import alc
filepath_base = 'salguero'


puntos= pd.read_csv(f'{filepath_base}/world_img_pts.csv', sep=';')
pts_3d= puntos[['X', 'Y', 'Z']]



correspondencias=pd.read_csv(f'{filepath_base}/gs_10_09_point_correspondences_v5.txt', sep=r'\s+', header=None).to_numpy()                  # Test para verificar que buildQ funcione, no se obtienen las proyecciones asi
indices = correspondencias[0, :]

puntos_3d = alc.traspuesta(correspondencias[1:4, :])
puntos_2d = alc.traspuesta(correspondencias[4:6, :])
#ejercicio 5.a:
"""hay 5 puntos, por el 1.a necesitamos 6 para obtener Q"""
#los parámetros intrínsecos de la cámara(K) son iguales
#para todas las vistas, pues son característicos de la cámara
#entonces para cada foto se tiene
#M1=K[R1 t1], M2=K[R2 t2], M3=K[R3 t3] M4=K[R4 t4]
#intento buscar M para cada foto, tendría infitas soluciones
#en particular la dimensión del núcleo es 2
def obtenerHsK_coplanar(imgPtsList, objPts, restringir_K12=False, restringir_K11_K22=False):
    """
    objPts los puntos U,V del mundo real
    imgPtsList los puntos en la imagen para cada vista
    """
    Hs = []
    for imgPts in imgPtsList:
        H = ej3.resolverDLT2D(imgPts, objPts)
        Hs.append(H)

    A = ej3.armarA(Hs)
    K = ej3.calcularMatrizK2D(A, restringir_K12, restringir_K11_K22)
    return Hs, K

def proyectarPuntosUVW(ptsObjList, K, R, t):
    Rt = np.hstack((R, t.reshape(-1,1)))
    ptsProyectados = []
    for ptsUVW in ptsObjList:
        ptUVW1 = np.hstack((ptsUVW, 1))
        ptXYZ = alc.multiplicar(Rt, ptUVW1)
        pt_uvlambd = alc.multiplicar(K, ptXYZ)
        pt_uv = pt_uvlambd / pt_uvlambd[2]
        ptsProyectados.append(pt_uv[:-1])

    return np.array(ptsProyectados)



#esto utilice para marcar los puntos a mano; no creo q vaya en el tp 

import os
import cv2
import csv


carpeta = "gs_dataset"

# Ordenamos las fotografías cronológicamente
fotos = sorted([
    nombre for nombre in os.listdir(carpeta)
    if nombre.lower().endswith(".jpg")
])

print("Cantidad de fotografías:", len(fotos))

# Los puntos que vamos a marcar
objetos = ["senora", "camioneta", "taxi"]

# Archivo donde guardaremos los resultados
archivo_salida = os.path.join(carpeta, "puntos_movimiento.csv")

resultados = []

# Variables utilizadas por el mouse
punto_actual = None
imagen_actual = None

def marcar(event, x, y, flags, param):
    global punto_actual

    if event == cv2.EVENT_LBUTTONDOWN:
        punto_actual = (x, y)


# Recorremos las fotografías
for numero, foto in enumerate(fotos):

    ruta = os.path.join(carpeta, foto)
    imagen = cv2.imread(ruta)

    if imagen is None:
        print("No se pudo abrir:", foto)
        continue

    print(f"\nFoto {numero + 1}/{len(fotos)}: {foto}")

    fila = [foto]
    imagen_marcada = imagen.copy()

    cv2.namedWindow("Marcar puntos", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Marcar puntos", marcar)

    for objeto in objetos:

        punto_actual = None

        while True:
            vista = imagen_marcada.copy()

            cv2.putText(
                vista,
                f"Foto {numero+1}/{len(fotos)} - Marcar: {objeto}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                vista,
                "CLICK: marcar | S: saltear | ESC: finalizar",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            cv2.imshow("Marcar puntos", vista)
            tecla = cv2.waitKey(20) & 0xFF

            if punto_actual is not None:
                u, v = punto_actual

                fila.extend([u, v])

                cv2.circle(
                    imagen_marcada,
                    (u, v),
                    6,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    imagen_marcada,
                    objeto,
                    (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                print(f"{objeto}: ({u}, {v})")
                break

            elif tecla == ord("s"):
                fila.extend(["", ""])
                print(f"{objeto}: no aparece")
                break

            elif tecla == 27:
                fila.extend(["", ""])
                print("Selección interrumpida")
                break

    resultados.append(fila)

    # Guardamos después de cada fotografía
    with open(archivo_salida, "w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            "foto",
            "u_senora", "v_senora",
            "u_camioneta", "v_camioneta",
            "u_taxi", "v_taxi"
        ])
        escritor.writerows(resultados)

    cv2.destroyAllWindows()

print("\nProceso terminado.")
print("Coordenadas guardadas en:", archivo_salida)

puntos = pd.read_csv("gs_dataset/puntos_movimiento.csv", sep=",")
print(puntos.head())