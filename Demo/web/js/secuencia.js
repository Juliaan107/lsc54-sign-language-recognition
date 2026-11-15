// Deja la seña recortada lo más parecida posible a los videos de entrenamiento antes de
// entregarla al modelo.
//
// 1. Huecos de las manos. Cuando MediaPipe pierde una mano durante unos cuadros, esos
//    cuadros quedan en cero y el modelo se confunde mucho (sin manos responde casi
//    siempre «Negro», «Por favor», «Verde» o «De nada»). Si la mano se ve antes y después
//    de un hueco corto, se reconstruye por interpolación. Solo huecos cortos: los largos
//    también existen en los videos de entrenamiento y rellenarlos empeora el resultado.
// 2. Ritmo. Los videos de entrenamiento tienen unos 18 cuadros por segundo reales. Un
//    equipo lento analiza 8 o 10 y uno rápido 30; la seña se lleva a 18 por segundo
//    usando la hora real de cada cuadro (repitiendo o saltando cuadros).

import { DIMENSION } from './puntos.js';

export const CUADROS_POR_SEGUNDO = 18;
export const HUECO_MAX_MS = 170;     // huecos de hasta 2 cuadros (a 18 por segundo)

const MANOS = [[99, 162], [162, 225]];                  // mano izquierda, mano derecha

function hayBloque(datos, fila, [desde, hasta]) {
  const base = fila * DIMENSION;
  for (let i = base + desde; i < base + hasta; i++) if (datos[i] !== 0) return true;
  return false;
}

/**
 * Rellena, en el mismo arreglo, los huecos cortos de cada mano.
 * @param {Float32Array} datos   T x 225
 * @param {number[]} tiempos     hora de cada cuadro, en ms
 * @returns {number} cuadros reconstruidos
 */
export function rellenarHuecos(datos, tiempos, huecoMaxMs = HUECO_MAX_MS) {
  const T = tiempos.length;
  let rellenados = 0;
  for (const bloque of MANOS) {
    let previo = -1;
    for (let f = 0; f < T; f++) {
      if (!hayBloque(datos, f, bloque)) continue;
      if (previo >= 0 && f - previo > 1 && tiempos[f] - tiempos[previo] <= huecoMaxMs) {
        for (let m = previo + 1; m < f; m++) {
          const peso = (tiempos[m] - tiempos[previo]) / (tiempos[f] - tiempos[previo]);
          for (let i = bloque[0]; i < bloque[1]; i++) {
            datos[m * DIMENSION + i] = datos[previo * DIMENSION + i] * (1 - peso) + datos[f * DIMENSION + i] * peso;
          }
          rellenados++;
        }
      }
      previo = f;
    }
  }
  return rellenados;
}

/**
 * Remuestrea la seña a un ritmo fijo: para cada instante toma el cuadro real más cercano.
 * (Interpolar entre cuadros suaviza el movimiento y, medido sobre el conjunto de prueba,
 * al modelo le va peor que con cuadros reales repetidos.)
 * @returns {{ datos: Float32Array, T: number }}
 */
export function remuestrear(datos, tiempos, porSegundo = CUADROS_POR_SEGUNDO) {
  const n = tiempos.length;
  const duracion = tiempos[n - 1] - tiempos[0];
  const T = Math.max(8, Math.round((duracion * porSegundo) / 1000) + 1);
  const salida = new Float32Array(T * DIMENSION);
  let a = 0;
  for (let f = 0; f < T; f++) {
    const t = tiempos[0] + (duracion * f) / (T - 1);
    while (a < n - 1 && Math.abs(tiempos[a + 1] - t) <= Math.abs(tiempos[a] - t)) a++;
    salida.set(datos.subarray(a * DIMENSION, (a + 1) * DIMENSION), f * DIMENSION);
  }
  return { datos: salida, T };
}

/** Aplica los dos pasos a una seña del segmentador. */
export function prepararSecuencia(datos, tiempos) {
  const { datos: salida, T } = remuestrear(datos, tiempos);
  const uniformes = Array.from({ length: T }, (_, f) => (f * 1000) / CUADROS_POR_SEGUNDO);
  const rellenados = rellenarHuecos(salida, uniformes);
  return { datos: salida, T, rellenados };
}
