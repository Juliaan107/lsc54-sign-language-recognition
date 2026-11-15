// Mano de la pantalla de inicio: cierra y abre los dedos despacio, pasando por varias
// formas sencillas. Solo se anima mientras la pantalla de inicio está a la vista.

// Cadenas de puntos de cada dedo (índices de los círculos del SVG): base y tres falanges.
const DEDOS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const PULGAR = [1, 2, 3, 4];
// Pulgar doblado sobre la palma.
const PULGAR_CERRADO = [[71, 168], [73, 147], [89, 139], [105, 137]];
const TRAZOS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12], [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [0, 17]];

// Cuánto se cierra cada dedo en cada forma: pulgar, índice, medio, anular, meñique.
const FORMAS = [
  [0, 0, 0, 0, 0],      // abierta
  [1, 0, 1, 1, 1],      // uno
  [1, 0, 0, 1, 1],      // dos
  [0, 0, 0, 1, 1],      // tres
  [1, 0, 0, 0, 0],      // cuatro
  [0, 0, 0, 0, 0],      // abierta
  [1, 1, 1, 1, 1],      // puño
  [0, 0, 1, 1, 1],      // ele
  [0, 0, 1, 1, 0],      // pulgar, índice y meñique
];
const PAUSA_MS = 2300, CAMBIO_MS = 1500, DESFASE_MS = 110;
const GRADOS = Math.PI / 180;

const suave = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

export function animarMano(svg, estaVisible) {
  if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const circulos = [...svg.querySelectorAll('.mano-puntos circle')];
  const trazos = [...svg.querySelectorAll('.mano-huesos path')];
  if (circulos.length !== 21 || trazos.length !== TRAZOS.length) return;
  const abierta = circulos.map((c) => [c.cx.baseVal.value, c.cy.baseVal.value]);
  const puntos = abierta.map((p) => p.slice());

  const colocar = (cierre) => {
    // Pulgar: se cruza sobre la palma.
    PULGAR.forEach((indice, k) => {
      const c = suave(cierre[0]);
      puntos[indice][0] = abierta[indice][0] + (PULGAR_CERRADO[k][0] - abierta[indice][0]) * c;
      puntos[indice][1] = abierta[indice][1] + (PULGAR_CERRADO[k][1] - abierta[indice][1]) * c;
    });
    // Los otros dedos se doblan hacia la palma: visto de frente, cada falange se acorta
    // y, al pasar de 90 grados, vuelve hacia abajo.
    DEDOS.forEach((dedo, d) => {
      const c = cierre[d + 1];
      let angulo = 0, x = abierta[dedo[0]][0], y = abierta[dedo[0]][1];
      [75, 100, 65].forEach((giro, k) => {
        angulo += giro * c * GRADOS;
        const a = abierta[dedo[k]], b = abierta[dedo[k + 1]];
        x += (b[0] - a[0]) * Math.cos(angulo);
        y += (b[1] - a[1]) * Math.cos(angulo);
        puntos[dedo[k + 1]][0] = x;
        puntos[dedo[k + 1]][1] = y;
      });
    });
    circulos.forEach((c, i) => {
      c.setAttribute('cx', puntos[i][0].toFixed(1));
      c.setAttribute('cy', puntos[i][1].toFixed(1));
    });
    trazos.forEach((t, i) => {
      t.setAttribute('d', 'M' + TRAZOS[i].map((p) => `${puntos[p][0].toFixed(1)} ${puntos[p][1].toFixed(1)}`).join(' '));
    });
  };

  const ciclo = PAUSA_MS + CAMBIO_MS;
  const cierre = [0, 0, 0, 0, 0];
  let inicio = null;
  const cuadro = (ahora) => {
    if (!estaVisible() || svg.getClientRects().length === 0) {
      inicio = null;
      setTimeout(() => requestAnimationFrame(cuadro), 700);
      return;
    }
    if (inicio === null) inicio = ahora;
    const t = ahora - inicio;
    const paso = Math.floor(t / ciclo);
    const desde = FORMAS[paso % FORMAS.length], hasta = FORMAS[(paso + 1) % FORMAS.length];
    const enCambio = (t % ciclo) - PAUSA_MS;
    for (let d = 0; d < 5; d++) {
      const avance = suave((enCambio - d * DESFASE_MS) / (CAMBIO_MS - 4 * DESFASE_MS));
      const respiro = 0.035 * (0.5 + 0.5 * Math.sin(ahora / 900 + d * 1.3));     // nunca del todo quieta
      cierre[d] = desde[d] + (hasta[d] - desde[d]) * avance + respiro;
    }
    colocar(cierre);
    requestAnimationFrame(cuadro);
  };
  requestAnimationFrame(cuadro);
}
