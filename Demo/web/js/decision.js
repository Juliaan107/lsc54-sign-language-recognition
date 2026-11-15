// Convierte la salida del modelo en una decisión para mostrar.
//
// El modelo devuelve dos filas de probabilidades: la seña tal como se vio y la misma
// seña reflejada horizontalmente. Por cada clase se toma la mayor de las dos; así el
// resultado no depende de la mano que use la persona ni de si la cámara voltea la imagen.

export const UMBRAL_CONFIANZA = 0.6;

// Señas que en los videos de entrenamiento casi nunca tienen la mano detectada, porque
// queda sobre la cara. Son las únicas que se aceptan cuando la página tampoco la vio;
// cualquier otra respuesta sin manos es una suposición del modelo y no se muestra.
export const SENAS_SIN_MANO = ['rosado', 'negro'];

/** ¿Se puede mostrar este resultado de una seña en la que casi no se vieron las manos? */
export function creibleSinManos(idClase) {
  return SENAS_SIN_MANO.includes(idClase);
}

/**
 * @param {Float32Array} probabilidades  2 x nClases, en orden de filas
 * @param {number} nClases
 * @returns {{ indice:number, confianza:number, reconocida:boolean,
 *             mejores:Array<{indice:number, confianza:number}> }}
 */
export function decidir(probabilidades, nClases, umbral = UMBRAL_CONFIANZA) {
  const puntajes = new Float32Array(nClases);
  for (let c = 0; c < nClases; c++) {
    puntajes[c] = Math.max(probabilidades[c], probabilidades[nClases + c]);
  }
  const orden = Array.from(puntajes.keys()).sort((a, b) => puntajes[b] - puntajes[a]);
  const mejores = orden.slice(0, 3).map((indice) => ({ indice, confianza: puntajes[indice] }));
  return {
    indice: mejores[0].indice,
    confianza: mejores[0].confianza,
    reconocida: mejores[0].confianza >= umbral,
    mejores,
  };
}
