# Demo: señas en vivo

Página web que reconoce en tiempo real las 54 señas del proyecto usando la cámara. No tiene
botones de grabar: detecta sola cuándo empieza y cuándo termina cada seña.

Todo corre en el navegador de quien la usa. No hay servidor que procese nada y el video no
sale del dispositivo; por eso se puede publicar como un sitio estático.

## Probarla en este computador

Desde la raíz del proyecto:

```
python Demo/herramientas/servidor_local.py
```

y abrir <http://localhost:8000>. Tiene que ser por `localhost` (o por https): el navegador
no permite usar la cámara si la página se abre con doble clic sobre `index.html`.

Agregar `?debug` a la dirección muestra datos técnicos sobre el video (análisis por
segundo, alturas de las muñecas y su reposo aprendido, estado del detector de señas, zoom).

## Publicarla (Hugging Face Spaces)

Solo se publica la carpeta `web/`. Una única vez hay que iniciar sesión y crear el espacio:

```
hf auth login
hf repo create senas-en-vivo --repo-type space --space-sdk static
```

Y cada vez que se quiera publicar o actualizar (desde la raíz del proyecto):

```
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='Demo/web', repo_id='Juliaan107/senas-en-vivo', repo_type='space')"
```

(`hf upload` no sirve aquí: intenta crear el Space otra vez y Hugging Face lo rechaza.)

La página publicada está en <https://juliaan107-senas-en-vivo.static.hf.space>. Conviene
compartir esa dirección y no la de `huggingface.co/spaces/...`: abre la página
directamente, sin el marco de Hugging Face alrededor.

## Cómo funciona

```
cámara ─▶ MediaPipe ─▶ 225 números por cuadro ─▶ segmentador ─▶ preparación ─▶ modelo ONNX ─▶ decisión
          (pose + manos,                          (¿empezó o     (ritmo y       (preprocesa
           con zoom a la persona)                  terminó?)      huecos)        y clasifica)
```

1. **Puntos** (`js/detector.js`, `js/puntos.js`). MediaPipe usa los mismos dos modelos con
   los que se construyó el dataset. `puntos.js` arma el vector de 225 números igual que
   `_frame_vector` de `lsc_pipeline.py`.
   - *Zoom a la persona.* Si la persona se ve pequeña (está lejos, o el celular está en
     vertical), se analiza solo un recuadro a su alrededor, con un encuadre parecido al de
     los videos de entrenamiento. Sin esto MediaPipe pierde las manos.
   - *Proporción del video.* Los videos de entrenamiento son 16:9. MediaPipe entrega las
     coordenadas como fracciones del ancho y del alto, así que un video vertical produce
     un esqueleto deformado (el modelo bajaba de 90 % a 61 % en la prueba). Antes de usar
     los puntos se llevan a un cuadro 16:9 (`corregirProporcion`).
2. **Segmentador** (`js/segmentador.js`). Aprende dónde descansan las manos de la persona
   (colgando, sobre las piernas, sobre la mesa, fuera de la imagen). La seña empieza cuando
   una muñeca sube desde ese reposo y termina cuando vuelve, o cuando las manos se quedan
   quietas y bajas en otro lugar. El recorte incluye un poco de reposo antes y después,
   como los videos de entrenamiento. Se descartan los movimientos muy cortos, los cambios
   de postura (subir la mano sin volver a bajarla) y las señas en las que casi no se
   vieron las manos.
3. **Preparación** (`js/secuencia.js`). La seña se lleva a 18 cuadros por segundo (el ritmo
   real de los videos de entrenamiento) repitiendo o saltando cuadros, y se reconstruyen
   los huecos cortos en los que MediaPipe perdió una mano.
4. **Modelo** (`modelos/lsc_tssi.onnx`). El archivo ONNX contiene todo el preprocesamiento
   del cuaderno (normalizar, imagen TSSI, escalado con `p_bajo` y `p_alto`, 128×128,
   ImageNet) y la DenseNet-121. El navegador solo le entrega la matriz cruda.
5. **Decisión** (`js/decision.js`). El modelo evalúa la seña en dos orientaciones (como se
   vio y reflejada) y se toma la más segura. Si la confianza es menor a 60 %, la página
   dice que no la reconoció.

### Por qué dos orientaciones

En el dataset los **colores** se grabaron con la mano hacia un lado de la imagen y los
**números y la cortesía** hacia el otro (en entrenamiento: 663 videos de colores con la
mano "izquierda" de MediaPipe contra 1 con la "derecha"). El modelo aprendió esa pista.
Evaluar las dos orientaciones hace que el demo no dependa de la mano que use la persona ni
de si la cámara voltea la imagen, y en el conjunto de prueba sube la exactitud de 87,1 % a
90,1 %.

La forma de fondo de corregirlo sería reentrenar con un aumento de reflejo horizontal, para
que el modelo deje de apoyarse en el lado de la mano. Hoy algunas parejas (por ejemplo
«Azul» y «Diez») quedan muy cerca entre sí.

### Qué pasa cuando no se ven las manos

Si a una seña se le borran las manos, el modelo acierta solo el 10 % y responde casi siempre
«Por favor», «Negro», «Verde» o «De nada», con confianza alta. Con la mano perdida en el
30 % de los cuadros baja de 90 % a 74 %. Por eso el demo cuida las manos en tres puntos:
el zoom (para que MediaPipe las vea), el relleno de huecos cortos (sube ese 74 % a 87 %) y
el descarte de las señas en las que casi no se vieron, con un aviso en pantalla.

Hay una excepción. En los videos de «Rosado» y «Negro» del dataset MediaPipe casi nunca
detectó la mano, porque queda sobre la cara: el modelo aprendió esas dos señas solo con el
cuerpo. Por eso, si no se vieron las manos pero la muñeca estuvo sobre la cara, la seña sí
se clasifica y se muestra únicamente cuando la respuesta es «Rosado» o «Negro»
(`creibleSinManos` en `js/decision.js`). Sin esta excepción el demo descartaba 10 de los 11
videos de prueba de «Rosado».

### Señas que el modelo no aprendió bien

`probar_flujo.mjs test … --porclase` muestra el resultado de cada seña. Las más débiles son
de cortesía y tienen pocos videos: «Sí» y «No» (el modelo acierta 2 de 6 con el video
completo; varios videos repiten la seña dos o tres veces seguidas, así que una sola
repetición no se parece a lo que vio en entrenamiento), «Gracias», «Muchas gracias» y
«Más o menos». «Gris» se hace con la mano muy baja y a veces no se detecta como seña.
Esto es del modelo y del dataset, no de la página.

## Qué se verificó

| Prueba | Resultado |
|---|---|
| ONNX contra PyTorch, 568 videos de prueba | mismas 568 predicciones; diferencia máxima de probabilidad 9e-6 |
| Exactitud en prueba, una orientación (como el cuaderno) | 87,1 % |
| Exactitud en prueba, dos orientaciones (como el demo) | 90,1 % |
| Detección automática sobre los videos de prueba | 95,2 % detectados como una sola seña; se muestra el 94,2 % y de esas acierta el 90,7 % |
| Lo mismo con la mitad de los cuadros (equipo lento) | se muestra el 93,5 % y acierta el 90,4 % |
| Lo mismo con las manos descansando sobre una mesa (simulado) | 94,0 % detectados; acierta el 89,7 % |
| Página completa en Edge con `video_ejemplo.mp4` | 6 de 6 señas (Buenos días, Diez, Chao, Ocho, Amarillo, Azul) |
| El mismo video convertido a vertical con la persona lejos | 6 de 6 (sin el zoom ni la corrección de proporción: 2 de 6) |

La prueba de la página se repitió en tamaño de escritorio y de celular, con y sin
aceleración gráfica, y con MediaPipe en un hilo aparte. No hay pruebas automáticas con un
celular real ni con el celular girado: eso se prueba a mano.

## Rendimiento

El límite lo pone MediaPipe (pose + manos). En un portátil con gráfica integrada (Ryzen 5
3500U, Radeon Vega 8) analiza unos 8 a 14 cuadros por segundo; clasificar cada seña tarda
alrededor de 0,25 s. Con una tarjeta gráfica mejor el análisis es más rápido. El video de
la cámara se ve fluido en cualquier caso. El esqueleto se dibuja en cada pintada de la
pantalla y se adelanta un poco según la velocidad de cada punto, para que no se vea
rezagado aunque lleguen pocos análisis por segundo.

Parámetros de la dirección para pruebas: `?debug` (datos técnicos), `?cpu`, `?hilo`
(MediaPipe en un worker), `?directo` (sin reducir el cuadro), `?sinzoom`.

## Carpetas

```
Demo/
├── web/                     ← el sitio (lo único que se publica)
│   ├── index.html, css/, js/, fuentes/
│   ├── modelos/             ← lsc_tssi.onnx, clases.json y los dos modelos de MediaPipe
│   └── vendor/              ← MediaPipe y ONNX Runtime Web (sin depender de internet externo)
├── herramientas/
│   ├── exportar_onnx.py     ← genera web/modelos/lsc_tssi.onnx y lo verifica
│   ├── servidor_local.py
│   └── pruebas/             ← pruebas del segmentador y de la página completa
└── video_ejemplo.mp4        ← video de prueba (no se publica)
```

## Si se reentrena el modelo

```
python Demo/herramientas/exportar_onnx.py
```

Lee `modelos/sl_tssi_aug/best.pt`, vuelve a escribir `web/modelos/lsc_tssi.onnx` y
`clases.json`, y comprueba que ONNX y PyTorch coincidan. Para usar otro modelo, cambiar
`RUTA_MODELO` al inicio de ese archivo.

## Volver a correr las pruebas

Las pruebas usan Node con `onnxruntime-node` y `playwright-core`:

```
cd Demo/herramientas/pruebas
npm install onnxruntime-node playwright-core
python preparar_datos.py datos ../../video_ejemplo.mp4
node probar_flujo.mjs video datos/video.npy 30
node probar_flujo.mjs test datos/flujo_test.npy datos/flujo_test.json 18
```

`probar_flujo.mjs test` acepta `--salto=2` (equipo lento), `--mesa=1.3` (manos apoyadas
en una mesa), `--cruda` (sin la preparación de la secuencia) y `--porclase` (resultado de
cada seña y con cuáles se confunde).

Para la página completa, con el servidor local iniciado con
`--video Demo/video_ejemplo.mp4`:

```
node probar_pagina.mjs --gpu
```

Con `--movil` usa tamaño de celular y con `--traza archivo.json` guarda lo que midió el
segmentador en cada cuadro.
