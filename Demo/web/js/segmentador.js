// Detecta sola dónde empieza y dónde termina cada seña, sin botones.
//
// Idea: los videos con los que se entrenó el modelo empiezan y terminan con las manos en
// reposo. Aquí se hace lo mismo con el flujo continuo de la cámara:
//
//   REPOSO ──(una mano sube desde su reposo)──▶ CAPTURANDO ──(vuelve al reposo)──▶ seña lista
//
// "Reposo" no es una altura fija: se aprende de la persona. Donde las manos se quedan
// quietas y bajas (colgando, sobre las piernas, sobre la mesa, fuera del cuadro), ese es
// su reposo. La seña empieza cuando una muñeca sube desde ahí y termina cuando vuelve, o
// cuando las manos se quedan quietas y bajas en otro lugar.
//
// Las alturas se miden desde los hombros, en "anchos de hombros", con las coordenadas ya
// llevadas a la proporción de los videos de entrenamiento (ver corregirProporcion en
// puntos.js). Así no dependen de la distancia a la cámara ni de si el video es vertical.
// Referencias medidas en el dataset: manos colgando ≈ 2.5, la mayoría de las señas llega
// por encima de 0.3 y casi ninguna se queda por debajo de 1.2.
//
// El recorte que se entrega al modelo imita a los videos de entrenamiento: además de la
// seña incluye un poco de reposo antes y después (30 % de su duración a cada lado).

import { DIMENSION } from './puntos.js';

export const OPCIONES = {
  alturaBase: 1.9,         // reposo que se supone mientras no se aprende el de la persona; el
                           // reposo aprendido nunca queda más abajo (ahí los puntos tiemblan)
  subir: 0.3,              // la seña empieza cuando una muñeca sube esto sobre su reposo
  volver: 0.15,            // y termina cuando vuelve a quedar a menos de esto de su reposo
  picoMinimo: 0.45,        // en algún momento la mano debe subir al menos esto sobre el reposo
  alturaReposoMin: 1.0,    // una mano quieta solo cuenta como "en reposo" por debajo de aquí
  radioQuieto: 0.2,        // cuánto puede moverse una mano y seguir contando como quieta
  suavizadoMs: 250,        // la quietud se mide sobre la posición suavizada (quita el temblor)
  quietudBaseMs: 350,      // tiempo quieta para aprender ahí el reposo
  quietudLibreMs: 1500,    // tras un bloqueo: quieta este tiempo en cualquier parte = reposo
  cuadrosParaIniciar: 2,   // cuadros seguidos con la mano arriba para empezar
  duracionMinMs: 350,      // movimientos más cortos se descartan
  cuadrosMin: 5,
  duracionMaxMs: 8000,     // más largo que esto no es una seña aislada
  fraccionEspera: 0.3,     // cuánto esperar en reposo antes de cerrar la seña…
  esperaMinMs: 400,        // …entre este mínimo
  esperaMaxMs: 700,        // …y este máximo
  fraccionReposo: 0.3,     // reposo que se añade antes y después, como fracción de la seña
  fraccionManosMin: 0.25,  // fracción mínima de cuadros de la seña con alguna mano detectada
  radioCara: 0.9,          // una muñeca a menos de esto de la nariz está "sobre la cara"
  fraccionCaraMin: 0.7,    // fracción de la seña sobre la cara para aceptar que no se vea la mano
  memoriaReposoMs: 2500,   // cuánto reposo previo se recuerda
  saltoMaxMs: 1000,        // si pasa más tiempo entre cuadros (pestaña oculta), se reinicia
};

const HOMBRO_IZQ = 11, HOMBRO_DER = 12, MUNECAS = [15, 16];
const INICIO_MANOS = 99;
const BLOQUE_MANO = [[99, 162], [162, 225]];      // mano de cada muñeca dentro del vector

function hayMano(v, i) {
  for (let k = BLOQUE_MANO[i][0]; k < BLOQUE_MANO[i][1]; k++) if (v[k] !== 0) return true;
  return false;
}

/**
 * Fracción de los cuadros con una mano levantada en los que esa mano está sobre la cara.
 * Ahí MediaPipe casi nunca detecta la mano: tampoco la detectó en los videos de
 * entrenamiento de «Rosado» y «Negro», que el modelo aprendió solo con el cuerpo.
 */
function fraccionSobreLaCara(tramo, radio) {
  let arriba = 0, cerca = 0;
  for (const { v } of tramo) {
    const ancho = Math.hypot(v[HOMBRO_IZQ * 3] - v[HOMBRO_DER * 3], v[HOMBRO_IZQ * 3 + 1] - v[HOMBRO_DER * 3 + 1]);
    if (!(ancho > 0)) continue;
    const cy = (v[HOMBRO_IZQ * 3 + 1] + v[HOMBRO_DER * 3 + 1]) / 2;
    let levantada = false, distancia = Infinity;
    for (const i of MUNECAS) {
      if ((v[i * 3 + 1] - cy) / ancho < 0.6) levantada = true;
      distancia = Math.min(distancia, Math.hypot(v[i * 3] - v[0], v[i * 3 + 1] - v[1]) / ancho);
    }
    if (!levantada) continue;
    arriba++;
    if (distancia < radio) cerca++;
  }
  return arriba ? cerca / arriba : 0;
}

function hayManos(v) {
  for (let i = INICIO_MANOS; i < DIMENSION; i++) if (v[i] !== 0) return true;
  return false;
}

/**
 * Mide la postura de un cuadro.
 * @param {Float32Array} v        vector de 225 números, ya en la proporción de entrenamiento
 * @param {number} factorX        factor que se usó para corregir la proporción (1 = ninguno)
 * @returns {{persona:boolean, ancho:number, borde:number, altura:number,
 *            alturas:number[], x:number[]}}
 *   alturas: de cada muñeca (Infinity si está fuera del cuadro); borde: altura del borde
 *   inferior del cuadro; x: posición horizontal de cada muñeca, en anchos de hombros.
 */
export function medir(v, factorX = 1) {
  const hix = v[HOMBRO_IZQ * 3], hiy = v[HOMBRO_IZQ * 3 + 1];
  const hdx = v[HOMBRO_DER * 3], hdy = v[HOMBRO_DER * 3 + 1];
  const ancho = Math.hypot(hix - hdx, hiy - hdy);
  if (!(ancho > 0.03 * factorX)) {
    return { persona: false, ancho: 0, borde: Infinity, altura: Infinity, alturas: [Infinity, Infinity], x: [0, 0] };
  }
  const cx = (hix + hdx) / 2, cy = (hiy + hdy) / 2;
  const alturas = [], xs = [];
  for (const indice of MUNECAS) {
    const x = v[indice * 3], y = v[indice * 3 + 1];
    const xCuadro = 0.5 + (x - 0.5) / factorX;          // posición en el cuadro real de la cámara
    const enCuadro = y < 1.0 && y > -0.5 && xCuadro > -0.2 && xCuadro < 1.2;
    alturas.push(enCuadro ? (y - cy) / ancho : Infinity);   // fuera del cuadro cuenta como "abajo"
    xs.push((x - cx) / ancho);
  }
  return { persona: true, ancho, borde: (1 - cy) / ancho, altura: Math.min(alturas[0], alturas[1]), alturas, x: xs };
}

export class Segmentador {
  constructor(opciones = {}) {
    this.op = { ...OPCIONES, ...opciones };
    this.factorX = 1;            // lo fija la página según la proporción del video
    this.reiniciar();
  }

  reiniciar() {
    this.estado = 'reposo';      // 'reposo' | 'capturando' | 'bloqueado'
    this.reposo = [];            // cuadros recientes en reposo
    this.cuadros = [];           // cuadros de la seña en curso
    this.racha = 0;
    this.tInicio = 0;
    this.tUltimoActivo = 0;
    this.tPrevio = null;
    this.tAnterior = 0;
    this.base = [this.op.alturaBase, this.op.alturaBase];   // altura de reposo de cada muñeca
    this.suave = [null, null];   // posición suavizada de cada muñeca
    this.ancla = [null, null];   // punto alrededor del cual cada muñeca está quieta
    this.tAncla = [0, 0];        // desde cuándo
    this.pico = [Infinity, Infinity];
    this.baseInicio = [0, 0];
    this.medida = medir(new Float32Array(DIMENSION));
  }

  /** Milisegundos que lleva la seña en curso (0 si no hay). */
  get duracionActual() {
    return this.estado === 'capturando' ? this.tPrevio - this.tInicio : 0;
  }

  /**
   * Procesa un cuadro. Devuelve null o un evento:
   *   { tipo: 'inicio' }
   *   { tipo: 'sena', datos: Float32Array, T, tInicio, tFin, duracionMs, pocasManos }
   *   { tipo: 'descartada', motivo: 'corta' | 'baja' | 'larga' | 'sin-manos' }
   */
  procesar(vector, t) {
    const op = this.op;
    if (this.tPrevio !== null && (t - this.tPrevio > op.saltoMaxMs || t < this.tPrevio)) this.reiniciar();
    this.tAnterior = this.tPrevio === null ? t : this.tPrevio;
    this.tPrevio = t;


    const m = medir(vector, this.factorX);
    this.medida = m;
    const capturando = this.estado === 'capturando';

    // Por cada muñeca: ¿está arriba de su reposo?, ¿desde cuándo está quieta?
    let arriba = false;            // alguna muñeca está por encima de su reposo
    let actividad = -Infinity;     // último instante en que alguna muñeca estaba haciendo algo
    let todasQuietas = true;
    for (let i = 0; i < 2; i++) {
      const a = m.alturas[i];
      const presente = Number.isFinite(a);
      if (!presente) {
        if (this.ancla[i] !== null || this.tAncla[i] === 0) { this.ancla[i] = null; this.tAncla[i] = t; }
        this.suave[i] = null;
      } else {
        // Cuando MediaPipe no ve bien la mano (colgando, lejos), el punto de la muñeca
        // tiembla sin que la mano se mueva. La posición suavizada no.
        const f = this.suave[i] ? 1 - Math.exp(-(t - this.tAnterior) / op.suavizadoMs) : 1;
        const q = this.suave[i] || { x: m.x[i], a };
        q.x += (m.x[i] - q.x) * f;
        q.a += (a - q.a) * f;
        this.suave[i] = q;
        const p = this.ancla[i];
        if (!p || Math.hypot(q.x - p.x, q.a - p.a) > op.radioQuieto) {
          this.ancla[i] = { x: q.x, a: q.a };
          this.tAncla[i] = t;
        }
      }
      const quietaMs = t - this.tAncla[i];
      const baja = !presente || a > op.alturaReposoMin;
      const limite = Math.min(this.base[i], m.borde, op.alturaBase);
      // En la zona baja la muñeca solo cuenta si MediaPipe además ve la mano: cuando no
      // la ve (está tapada o muy lejos), la posición de la muñeca es una suposición.
      const estaArriba = presente && a < limite - (capturando ? op.volver : op.subir) &&
        (!baja || hayMano(vector, i));

      if (estaArriba) {
        arriba = true;
        // Una mano quieta y baja ya no está haciendo nada, aunque no haya vuelto a su reposo.
        actividad = Math.max(actividad, baja ? this.tAncla[i] : t);
        if (a < this.pico[i]) this.pico[i] = a;
      }
      if (presente && quietaMs < op.quietudLibreMs) todasQuietas = false;

      // Aprender el reposo: fuera de una seña, donde la mano se queda quieta y baja.
      if (!capturando && !estaArriba && quietaMs >= op.quietudBaseMs) {
        // Se acerca poco a poco, para quedarse con el centro del temblor y no con un extremo.
        const destino = presente ? this.suave[i].a : op.alturaBase;
        this.base[i] = quietaMs - (t - this.tAnterior) < op.quietudBaseMs ? destino : this.base[i] + 0.2 * (destino - this.base[i]);
      }
    }
    const cuadro = { v: vector, t };

    if (this.estado === 'reposo') {
      this.reposo.push(cuadro);
      this.racha = arriba ? this.racha + 1 : 0;
      if (this.racha >= op.cuadrosParaIniciar) {
        // Los últimos cuadros de la racha ya son parte de la seña.
        this.cuadros = this.reposo.splice(this.reposo.length - this.racha);
        this.tInicio = this.cuadros[0].t;
        this.tUltimoActivo = t;
        this.baseInicio = [0, 1].map((i) => Math.min(this.base[i], m.borde, op.alturaBase));
        this.racha = 0;
        this.estado = 'capturando';
        return { tipo: 'inicio' };
      }
      if (!arriba) this.pico = [Infinity, Infinity];
      while (this.reposo.length && t - this.reposo[0].t > op.memoriaReposoMs) this.reposo.shift();
      return null;
    }

    if (this.estado === 'capturando') {
      this.cuadros.push(cuadro);
      if (actividad > this.tUltimoActivo) this.tUltimoActivo = actividad;
      const duracion = this.tUltimoActivo - this.tInicio;
      const espera = Math.min(op.esperaMaxMs, Math.max(op.esperaMinMs, op.fraccionEspera * duracion));
      if (t - this.tUltimoActivo >= espera) return this.cerrar(m);
      if (t - this.tInicio > op.duracionMaxMs) {
        this.estado = 'bloqueado';
        this.cuadros = [];
        this.reposo = [];
        this.pico = [Infinity, Infinity];
        return { tipo: 'descartada', motivo: 'larga' };
      }
      return null;
    }

    // 'bloqueado': la mano lleva demasiado tiempo arriba. Se sale cuando baja, o cuando la
    // persona se queda quieta un buen rato (entonces esa postura pasa a ser su reposo).
    if (actividad > this.tUltimoActivo) this.tUltimoActivo = actividad;
    if (todasQuietas) {
      for (let i = 0; i < 2; i++) if (this.ancla[i]) this.base[i] = this.ancla[i].a;
      this.aReposo(cuadro);
    } else if (t - this.tUltimoActivo >= op.esperaMinMs) {
      this.aReposo(cuadro);
    }
    return null;
  }

  aReposo(cuadro) {
    this.estado = 'reposo';
    this.reposo = [cuadro];
    this.racha = 0;
    this.pico = [Infinity, Infinity];
  }

  cerrar(m) {
    const op = this.op;
    const tramo = this.cuadros;
    let n = 0;                                       // cuadros con la seña
    while (n < tramo.length && tramo[n].t <= this.tUltimoActivo) n++;
    const duracionMs = this.tUltimoActivo - this.tInicio;
    const cola = tramo.slice(n);                     // reposo posterior ya observado
    const antes = this.reposo;
    const pico = this.pico;

    // El lugar donde quedaron las manos es su nuevo reposo.
    for (let i = 0; i < 2; i++) {
      if (!this.ancla[i]) this.base[i] = op.alturaBase;
      else if (this.ancla[i].a > op.alturaReposoMin) this.base[i] = this.ancla[i].a;
    }
    const baseFin = [0, 1].map((i) => Math.min(this.base[i], m.borde, op.alturaBase));

    this.estado = 'reposo';
    this.reposo = cola.slice();                      // el reposo de la siguiente seña empieza aquí
    this.cuadros = [];
    this.racha = 0;
    this.pico = [Infinity, Infinity];

    if (duracionMs < op.duracionMinMs || n < op.cuadrosMin) return { tipo: 'descartada', motivo: 'corta' };
    // Una seña sube y vuelve a bajar. Si la mano solo cambió de postura (por ejemplo, de
    // colgar a apoyarse sobre la mesa), no llega a estar bien por encima de ambos reposos.
    const subio = [0, 1].some((i) => pico[i] < Math.min(this.baseInicio[i], baseFin[i]) - op.picoMinimo);
    if (!subio) return { tipo: 'descartada', motivo: 'baja' };
    // Si casi no se vieron las manos, la seña solo sigue cuando la mano estuvo sobre la
    // cara; la página decide después si el resultado es creíble (ver decision.js).
    let pocasManos = false;
    if (op.fraccionManosMin > 0) {
      let conManos = 0;
      for (let i = 0; i < n; i++) if (hayManos(tramo[i].v)) conManos++;
      pocasManos = conManos < op.fraccionManosMin * n;
      if (pocasManos && fraccionSobreLaCara(tramo, op.radioCara) < op.fraccionCaraMin) {
        return { tipo: 'descartada', motivo: 'sin-manos' };
      }
    }

    const relleno = Math.round(op.fraccionReposo * n);
    const previos = antes.slice(Math.max(0, antes.length - relleno));
    const posteriores = cola.slice(0, relleno);
    const primero = previos.length ? previos[0] : tramo[0];
    const ultimo = posteriores.length ? posteriores[posteriores.length - 1] : tramo[n - 1];

    const T = relleno + n + relleno;
    const datos = new Float32Array(T * DIMENSION);
    const tiempos = [];                              // hora de cada cuadro entregado
    const paso = duracionMs / Math.max(1, n - 1);
    let fila = 0;
    const agregar = (c, t) => { datos.set(c.v, fila * DIMENSION); tiempos.push(t); fila++; };
    // Si falta reposo real, se repite el cuadro más cercano (con horas inventadas al mismo ritmo).
    for (let i = relleno - previos.length; i > 0; i--) agregar(primero, primero.t - i * paso);
    for (const c of previos) agregar(c, c.t);
    for (let i = 0; i < n; i++) agregar(tramo[i], tramo[i].t);
    for (const c of posteriores) agregar(c, c.t);
    for (let i = 1; i <= relleno - posteriores.length; i++) agregar(ultimo, ultimo.t + i * paso);

    return { tipo: 'sena', datos, tiempos, T, tInicio: this.tInicio, tFin: this.tUltimoActivo, duracionMs, pocasManos };
  }
}
