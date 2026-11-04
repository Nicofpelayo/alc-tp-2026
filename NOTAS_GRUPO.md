# Notas para la entrega

## Carpetas y rutas
- Seguimos trabajando con `ejercicios/alc.py`.
- Al armar la entrega, **copiamos `alc.py` a `template-alumnos/`**, al lado de `grupo_XX_TP.ipynb`. Así `import alc` anda sin tocar `sys.path`.
- En la notebook final, rutas relativas a `template-alumnos/`, como en la plantilla: `'futbol/test-gol-enzo.png'`, `'calib/coplanar/p_corners.txt'`. **No** usar `'../template-alumnos/...'`, porque en la máquina del docente no va a existir.

## alc.py
- Cada función con encabezado: qué parámetros recibe y qué devuelve. Lo pide el enunciado.
- Sin funciones repetidas. Por ejemplo, `indice_min` y `minimizacion` están en `ejercicio1.py` y en `ejercicio2.py`, y `buildQ` / `obtenerQ3d` hacen lo mismo.
- Dentro de `alc.py` se llama a las funciones directamente (`inversa(...)`, no `alc.inversa(...)`).

## Funciones permitidas
Es una **lista cerrada** (pág. 8-9): lo que no está, no se puede usar. Hoy hay estas cosas fuera de la lista:

| Dónde | Qué |
|---|---|
| `ejercicios/alc.py:139` (`rota`) | `np.cos`, `np.sin` |
| `ejercicios/alc.py:291` (`normaMatMC`) | `A@x` |
| `ejercicios/alc.py:305` y `:316` (`condMC`, `condExacto`) | `np.linalg.inv` |
| `ejercicios/ejercicio6.py:259-260` (prints de prueba) | `R.T @ R`, `np.linalg.det` |
| `ejercicios/ejercicio5.ipynb` | `cv2.cvtColor`, `cv2.circle` |

Para verificar durante el desarrollo se pueden usar, pero remover para la entrega

Imágenes: usamos `cv2.imread`. Devuelve BGR, así que para mostrarla con `plt.imshow` usar `img[:, :, ::-1]` (no `cv2.cvtColor`, que no está permitida).

## grupo_XX_TP.ipynb
Lo que hice del ej 4 (`ejercicios/ejercicio4.py` y `ejercicio4.ipynb`). Revisen si les parece bien o si hay que cambiar algo antes de pasarlo a la entrega:

- **4.b**: usé la misma H del 4.a. Los círculos se arman sin `np.cos`/`np.sin`, con $V = V_0 \pm \sqrt{r^2 - (U-U_0)^2}$. Los pies de Romero están tapados; su posición está estimada por la bota.
- **4.c**:
  - Calculé una H nueva con 7 puntos de esta foto.
  - Último defensor: el segundo de mayor U contando al arquero.
  - Resultado: hay offside. El que está a la derecha del 24 está adelantado +0.32m y el 25, +0.19m. Es muy justo, del orden del error de marcar los pies a mano, y el pie del de la derecha del 24 parece estar en el aire.
- **Validaciones visuales**: proyecto con H puntos o líneas que no se usaron para calcularla y los comparo con la foto: en el 3, las 77 esquinas del tablero; en el 4.a, el área chica; en el 4.c, líneas de la cancha. El enunciado no las pide, así que por ahora están solo en los archivos de trabajo (`ejercicio3.ipynb`, `ejercicio4.py`, `ejercicio4.ipynb`) y no en la entrega. ¿Las sumamos a la entrega?
- **4.a**: los puntos 5 y 6 usan el radio de la medialuna (9.15m), que sale del reglamento y no figura en el plano. Hay que consultar si vale... no me parece mal 

## Dudas para consultar
- **Ej 4.a**: usamos el radio de la medialuna (9.15 m, del reglamento) que no figura como número en el plano. ¿Vale?

## Estado (4/11)
| Ej | Trabajo (`ejercicios/`) | En la entrega |
|---|---|---|
| 1 | hecho | texto hecho |
| 2 | hecho | texto del 2.c a emprolijar |
| 3 | hecho | pasado a `alc.py` y a la notebook, en revisión |
| 4 | hecho (a, b, c), en revisión | pasado a `alc.py` y a la notebook, en revisión |
| 5 | recién empezado | pendiente |
| 6 | `.py` casi completo, notebook parcial | pendiente |
