// MediaPipe (pose y manos) corriendo en un hilo aparte, para que la página no se congele
// mientras se analiza un cuadro.
//
// Los dos modelos viven en el mismo worker a propósito: así comparten un solo contexto
// gráfico. Con un worker por modelo, la tarjeta gráfica pierde tiempo alternando entre
// contextos y el análisis sale casi el doble de lento en equipos con gráfica integrada.

import { FilesetResolver, PoseLandmarker, HandLandmarker } from '../vendor/mediapipe/vision_bundle.mjs';

const RUTA_WASM = new URL('../vendor/mediapipe/wasm', import.meta.url).href;

let pose = null;
let manos = null;
let ultimoTiempo = 0;

/** Empaqueta listas de puntos como números seguidos: x, y, z, visibilidad. */
function empaquetar(listas) {
  const total = listas.reduce((s, l) => s + l.length, 0);
  const datos = new Float32Array(total * 4);
  let k = 0;
  for (const lista of listas) {
    for (const p of lista) {
      datos[k++] = p.x;
      datos[k++] = p.y;
      datos[k++] = p.z;
      datos[k++] = p.visibility === undefined ? 1 : p.visibility;
    }
  }
  return datos;
}

async function cargar(m) {
  const archivos = await FilesetResolver.forVisionTasks(RUTA_WASM, true);
  // MediaPipe consume su "fábrica" al crear un modelo y, cargado como módulo, no la
  // repone para el segundo. Se guarda aquí y se restaura antes de cada creación.
  const { default: fabrica } = await import(archivos.wasmLoaderPath);
  self.ModuleFactory = fabrica;
  pose = await PoseLandmarker.createFromOptions(archivos, {
    baseOptions: { modelAssetBuffer: new Uint8Array(m.modeloPose), delegate: m.delegado },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
  self.ModuleFactory = fabrica;
  manos = await HandLandmarker.createFromOptions(archivos, {
    baseOptions: { modelAssetBuffer: new Uint8Array(m.modeloManos), delegate: m.delegado },
    runningMode: 'VIDEO',
    numHands: 2,
  });

  // El primer análisis siempre tarda (compila en la tarjeta gráfica); se hace por adelantado.
  const lienzo = new OffscreenCanvas(640, 360);
  const ctx = lienzo.getContext('2d');
  ctx.fillStyle = '#777';
  ctx.fillRect(0, 0, 640, 360);
  for (let i = 0; i < 3; i++) {
    pose.detectForVideo(lienzo, ++ultimoTiempo);
    manos.detectForVideo(lienzo, ultimoTiempo);
  }
}

self.onmessage = async (mensaje) => {
  const m = mensaje.data;
  try {
    if (m.tipo === 'cargar') {
      await cargar(m);
      self.postMessage({ tipo: 'listo' });
    } else if (m.tipo === 'detectar') {
      // MediaPipe exige marcas de tiempo estrictamente crecientes.
      ultimoTiempo = Math.max(m.t, ultimoTiempo + 1);
      const rPose = pose.detectForVideo(m.imagen, ultimoTiempo);
      const rManos = manos.detectForVideo(m.imagen, ultimoTiempo);
      m.imagen.close();
      const listasPose = rPose.landmarks || [], listasManos = rManos.landmarks || [];
      const datosPose = empaquetar(listasPose), datosManos = empaquetar(listasManos);
      self.postMessage({
        tipo: 'resultado', id: m.id,
        pose: datosPose, cantidadPose: listasPose.length,
        manos: datosManos, cantidadManos: listasManos.length,
      }, [datosPose.buffer, datosManos.buffer]);
    }
  } catch (e) {
    if (m.imagen && m.imagen.close) m.imagen.close();
    self.postMessage({ tipo: 'error', id: m.id, mensaje: String(e && e.message ? e.message : e) });
  }
};
