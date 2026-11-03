# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:48:57 2026

@author: anaba
"""

import ejercicio2 as ej2
import pandas as pd
import alc
filepath_base = '../template-alumnos/salguero'


puntos= pd.read_csv(f'{filepath_base}/world_img_pts.csv', sep=';')
pts_3d= puntos[['X', 'Y', 'Z']]


correspondencias=pd.read_csv(f'{filepath_base}/gs_10_09_point_correspondences_v5.txt', sep=r'\s+', header=None).to_numpy()                  # Test para verificar que buildQ funcione, no se obtienen las proyecciones asi

#ejercicio 5.a:
"""hay 5 puntos, por el 1.a necesitamos 6 para obtener Q"""
#los parámetros intrínsecos de la cámara(K) son iguales
#para todas las vistas, pues son característicos de la cámara
#entonces para cada foto se tiene
#M1=K[R1 t1], M2=K[R2 t2], M3=K[R3 t3] M4=K[R4 t4]
#intento buscar M para cada foto, tendría infitas soluciones
#en particular la dimensión del núcleo es 2
def par_intrinsecos_camara(pts_3d,pts_img):
    matrices=[]
    for i in range(len(pts_img)): 
        matriz_M=ej2.resolverDLT3D(ej2.obtenerQ3d(pts_img, pts_3d[i])) 
        matrices.append(matriz_M)
    return matrices


