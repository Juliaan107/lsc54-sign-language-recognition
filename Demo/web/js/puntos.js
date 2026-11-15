// Convierte los resultados de MediaPipe en el vector de 225 números con el que se
// entrenó el modelo. Es una traducción directa de `_frame_vector` (lsc_pipeline.py):
//
//   pose (33 puntos x 3) + mano izquierda (21 x 3) + mano derecha (21 x 3) = 225
//
// Los puntos que MediaPipe no detecta quedan en cero.

export const DIMENSION = 225;

const POSE = 0;
const MANO_IZQ = 99;
const MANO_DER = 162;
const MUNECA_IZQ = 15;
const MUNECA_DER = 16;

function copiar(destino, desplazamiento, puntos) {
  for (let i = 0; i < puntos.length; i++) {
    const p = puntos[i];
    destino[desplazamiento + i * 3] = p.x;
    destino[desplazamiento + i * 3 + 1] = p.y;
    destino[desplazamiento + i * 3 + 2] = p.z;
  }
}

/**
 * @param {Array<{x:number,y:number,z:number}>|null} pose  33 puntos normalizados, o null
 * @param {Array<Array<{x:number,y:number,z:number}>>|null} manos  hasta dos manos de 21 puntos
 * @returns {Float32Array} vector de 225 números
 */
export function vectorDeCuadro(pose, manos) {
  const v = new Float32Array(DIMENSION);
  if (pose && pose.length >= 33) copiar(v, POSE, pose);
  if (!manos || manos.length === 0) return v;

  // Cada mano se asigna a la muñeca del cuerpo que le queda más cerca.
  const ix = v[MUNECA_IZQ * 3], iy = v[MUNECA_IZQ * 3 + 1];
  const dx = v[MUNECA_DER * 3], dy = v[MUNECA_DER * 3 + 1];
  const sinPose = Math.abs(ix) < 1e-8 && Math.abs(iy) < 1e-8 &&
                  Math.abs(dx) < 1e-8 && Math.abs(dy) < 1e-8;

  let izquierdaUsada = false, derechaUsada = false;
  for (const mano of manos) {
    if (!mano || mano.length < 21) continue;
    let esIzquierda;
    if (sinPose) {
      esIzquierda = true;
    } else {
      const mx = mano[0].x, my = mano[0].y;
      esIzquierda = Math.hypot(mx - ix, my - iy) <= Math.hypot(mx - dx, my - dy);
    }
    if (esIzquierda ? izquierdaUsada : derechaUsada) esIzquierda = !esIzquierda;
    if (esIzquierda) {
      izquierdaUsada = true;
      copiar(v, MANO_IZQ, mano);
    } else {
      derechaUsada = true;
      copiar(v, MANO_DER, mano);
    }
  }
  return v;
}

/** Indica a qué lado quedó asignada cada mano detectada (para dibujarlas con su color). */
export function ladosDeManos(vector) {
  const hay = (desde) => {
    for (let i = desde; i < desde + 63; i++) if (vector[i] !== 0) return true;
    return false;
  };
  return { izquierda: hay(MANO_IZQ), derecha: hay(MANO_DER) };
}

// Los videos de entrenamiento son horizontales (16:9). MediaPipe entrega x e y como
// fracciones del ancho y del alto del cuadro, así que la misma persona produce números
// distintos si el video es vertical (un celular) o 4:3. Aquí se llevan los puntos a un
// cuadro 16:9 imaginario con la misma altura: equivale a añadirle franjas a los lados.
export const PROPORCION_ENTRENAMIENTO = 16 / 9;

/** Factor por el que hay que multiplicar las distancias horizontales. */
export function factorDeProporcion(anchoVideo, altoVideo) {
  if (!anchoVideo || !altoVideo) return 1;
  const factor = anchoVideo / altoVideo / PROPORCION_ENTRENAMIENTO;
  return Math.abs(factor - 1) < 0.02 ? 1 : factor;
}

/**
 * @param {Float32Array} vector  225 números tal como salen de MediaPipe
 * @param {number} factor        ver factorDeProporcion
 * @returns {Float32Array} el mismo vector si no hay nada que corregir, o una copia corregida
 */
export function corregirProporcion(vector, factor) {
  if (factor === 1) return vector;
  const v = new Float32Array(vector);
  for (const [desde, hasta] of [[POSE, MANO_IZQ], [MANO_IZQ, MANO_DER], [MANO_DER, DIMENSION]]) {
    let hay = false;
    for (let i = desde; i < hasta; i++) if (v[i] !== 0) { hay = true; break; }
    if (!hay) continue;                               // lo que no se detectó sigue en cero
    for (let i = desde; i < hasta; i += 3) {
      v[i] = 0.5 + (v[i] - 0.5) * factor;             // x
      v[i + 2] *= factor;                             // z usa la misma escala que x
    }
  }
  return v;
}
