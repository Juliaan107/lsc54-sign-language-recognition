---
title: Señas en vivo
emoji: 🤟
colorFrom: green
colorTo: indigo
sdk: static
pinned: false
short_description: Reconoce 54 señas de la Lengua de Señas Colombiana
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Señas en vivo

Reconoce 54 señas de la Lengua de Señas Colombiana (colores, números y cortesía) con la
cámara, en tiempo real y sin botones: la página detecta sola cuándo empieza y cuándo
termina cada seña.

Todo se ejecuta en el navegador. El video nunca sale del dispositivo.

- **Puntos del cuerpo:** MediaPipe (pose y manos), 225 números por cuadro.
- **Modelo:** SL-TSSI con DenseNet-121, entrenado con aumento de datos y exportado a ONNX.
- **Ejecución:** ONNX Runtime Web (WebAssembly).

Para usarla hay que permitir el acceso a la cámara. Funciona mejor de la cintura hacia
arriba, con buena luz y bajando las manos entre una seña y la siguiente.
