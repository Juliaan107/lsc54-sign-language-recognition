// Corre el modelo ONNX fuera del hilo principal, para que la cámara y el esqueleto
// no se congelen mientras se clasifica una seña.

import * as ort from '../vendor/ort/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
// Varios hilos solo están disponibles si la página está aislada (cabeceras COOP/COEP).
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1))
  : 1;

let sesion = null;

async function ejecutar(datos, T) {
  const entrada = new ort.Tensor('float32', datos, [T, 225]);
  const inicio = performance.now();
  const salida = await sesion.run({ secuencia: entrada });
  return {
    probabilidades: salida.probabilidades.data,
    imagen: salida.imagen.data,
    ms: performance.now() - inicio,
  };
}

self.onmessage = async (mensaje) => {
  const m = mensaje.data;
  try {
    if (m.tipo === 'cargar') {
      sesion = await ort.InferenceSession.create(new Uint8Array(m.modelo), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      // Una pasada de calentamiento: la primera siempre es la más lenta.
      const calentamiento = await ejecutar(new Float32Array(30 * 225).fill(0.5), 30);
      self.postMessage({ tipo: 'listo', hilos: ort.env.wasm.numThreads, ms: calentamiento.ms });
    } else if (m.tipo === 'clasificar') {
      const r = await ejecutar(m.datos, m.T);
      self.postMessage({ tipo: 'resultado', id: m.id, ...r });
    }
  } catch (e) {
    self.postMessage({ tipo: 'error', id: m.id, mensaje: String(e && e.message ? e.message : e) });
  }
};
