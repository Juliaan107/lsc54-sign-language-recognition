# Reconocimiento de Lengua de Señas Colombiana (LSC-54)

Proyecto final de Inteligencia Artificial II.

Nicolas Echeverri Rueda · Joseph Felipe Rodriguez · Julian Delgado Guerrero

El sistema identifica cuál de **54 señas aisladas** de la Lengua de Señas Colombiana aparece
en un video: 11 colores, 10 números y 33 expresiones de cortesía. Cada video se convierte con
MediaPipe en una secuencia de puntos del cuerpo y las manos (225 valores por fotograma) y un
modelo clasifica la seña.

**Demo en vivo:** <https://juliaan107-senas-en-vivo.static.hf.space>. Reconoce las señas con
la cámara, en el navegador. Ver [Demo/LEEME.md](Demo/LEEME.md).

## Resultados

2.880 videos de 34 personas, divididos **por persona** en 1.733 / 579 / 568 videos
(train / validación / prueba). Ninguna persona de prueba aparece en entrenamiento.

| Modelo                                            | Parámetros | Exactitud en prueba | F1 macro   |
| ------------------------------------------------- | ---------- | ------------------- | ---------- |
| Transformer de signjoey, coordenadas crudas       | 22,3 M     | 51,2 %              | 51,4 %     |
| Transformer de signjoey, coordenadas normalizadas | 22,3 M     | 65,1 %              | 65,1 %     |
| SL-TSSI-DenseNet                                  | 7,0 M      | 84,7 %              | 82,0 %     |
| **SL-TSSI-DenseNet con aumento de datos**         | 7,0 M      | **87,1 %**          | **85,2 %** |

- **Normalizar las coordenadas** (centrar el cuerpo en los hombros y cada mano en su muñeca)
  fue la mejora más importante para los modelos de secuencia.
- **SL-TSSI-DenseNet** convierte la secuencia en una imagen y la clasifica con una DenseNet-121
  preentrenada en ImageNet.
- **El aumento de datos** se aplica al vuelo sobre las coordenadas de train. La mejora frente
  al modelo sin aumento no es concluyente con una sola corrida (McNemar, p = 0,076).

El análisis completo, con limitaciones y trabajo futuro, está en
[Proyecto_LSC54.ipynb](Proyecto_LSC54.ipynb).

## Contenido

```
├── Proyecto_LSC54.ipynb      cuaderno final del proyecto
├── Desarrollo_LSC54.ipynb    cuaderno de trabajo: el proceso completo, paso a paso
├── lsc_pipeline.py           extracción con MediaPipe, división y entrenamiento de signjoey
├── tssi_aumento.py           SL-TSSI-DenseNet con aumento de datos
├── requirements.txt
├── slt/                      signjoey (Camgöz et al.), con ajustes para Windows y configs LSC
├── mediapipe_models/         modelos de pose y manos de MediaPipe
├── dataset_completo/         split.json y manifest.json (las características no se incluyen)
├── dataset_simple/           dataset de 5 videos de la primera prueba
├── modelos/                  pesos y resultados de cada modelo
└── Demo/                     página web en tiempo real (web/) y sus herramientas
```

## Instalación

Probado con Python 3.13.5 en Windows, solo con CPU.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Qué no está en el repositorio

Por tamaño, estos archivos no se suben:

| Qué                                      | Dónde iría                                               | Cómo se obtiene                                                               |
| ---------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Videos originales (947 MB)               | `Data/RawVideos/`                                        | En: https://www.scidb.cn/en/detail?dataSetId=0fa5acc6293543bba9a39a822988843e |
| Características extraídas                | `dataset_completo/lsc*.train/dev/test/all` y `features/` | `python lsc_pipeline.py dataset --workers 4` (~1 h, necesita los videos)      |
| Checkpoints del Transformer (288 MB c/u) | `modelos/lsc_full/`, `modelos/lsc_norm/`                 | Reentrenando (~5 h cada uno)                                                  |

Los videos siguen esta estructura:

```
Data/RawVideos/
├── Videos Colors/<señalante>/Colores/<seña>/<n>.avi
├── Videos Courtesy/<señalante>/Cordialidad/<seña>/<n>.avi
└── Videos Numbers/<señalante>/Numeros/<seña>/<n>.avi
```

Los resultados del Transformer (predicciones, registros de entrenamiento y métricas) sí están
incluidos, al igual que los pesos de SL-TSSI-DenseNet y de la LSTM.

## Cómo ejecutar

Los cuadernos ya están ejecutados y muestran todos los resultados.

Para volver a ejecutar `Proyecto_LSC54.ipynb`, ábrelo desde la raíz del proyecto y córrelo de
arriba abajo. Cada etapa revisa si su resultado ya está en disco y, si está, lo carga. Lo que
falte se vuelve a generar al llegar a su sección, y eso puede tardar horas:

| Etapa                        | Qué revisa                             | Si falta                   |
| ---------------------------- | -------------------------------------- | -------------------------- |
| Extracción con MediaPipe     | `dataset_completo/lsc.all`             | ~1 h (necesita los videos) |
| LSTM de diagnóstico          | `modelos/lstm/*.pt`                    | ~3,5 h                     |
| Transformer                  | `modelos/lsc_full`, `modelos/lsc_norm` | ~5 h cada uno              |
| SL-TSSI-DenseNet             | `modelos/sl_tssi_v2/best.pt`           | ~2 h                       |
| SL-TSSI-DenseNet con aumento | `modelos/sl_tssi_aug/best.pt`          | ~3 h                       |

Con un clon de este repositorio faltan las características y los checkpoints del Transformer
(ver la sección anterior), así que el cuaderno intentará regenerarlos.

Para probar el demo en este computador:

```
python Demo/herramientas/servidor_local.py
```

y abrir <http://localhost:8000>.

## Referencias

- N. C. Camgöz, O. Koller, S. Hadfield y R. Bowden. _Sign Language Transformers: Joint
  End-to-end Sign Language Recognition and Translation_. CVPR 2020.
  Código: <https://github.com/neccam/slt> (licencia Apache 2.0, ver `slt/LICENSE`).
- D. Laines, M. Gonzalez-Mendoza, G. Ochoa-Ruiz y G. Bejarano. _Isolated Sign Language
  Recognition based on Tree Structure Skeleton Images_. CVPRW 2023.
- Modelos de MediaPipe tomados de <https://github.com/juanesmz/SignCapture>.
