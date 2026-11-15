// Página del demo: une la cámara, MediaPipe, el segmentador y el clasificador.
//
//   cámara → Detector (puntos) → Segmentador (¿empezó o terminó una seña?)
//          → Clasificador (modelo ONNX) → decisión → pantalla
//
// Parámetros de la dirección (para pruebas):
//   ?debug            muestra datos técnicos sobre el video
//   ?video=<ruta>     usa un archivo de video en lugar de la cámara
//   ?cpu              fuerza MediaPipe sin aceleración gráfica
//   ?hilo             corre MediaPipe en un hilo aparte (ver detector.js)
//   ?directo          analiza el video a tamaño completo, sin reducirlo antes
//   ?sinzoom          no acerca el análisis a la persona cuando se ve pequeña

import { Detector } from './detector.js';
import { Clasificador } from './clasificador.js';
import { Segmentador } from './segmentador.js';
import { creibleSinManos, decidir, UMBRAL_CONFIANZA } from './decision.js';
import { pintarCinta } from './cinta.js';
import { Esqueleto, Seguidor } from './esqueleto.js';
import { animarMano } from './mano.js';
import { corregirProporcion, factorDeProporcion } from './puntos.js';
import { prepararSecuencia } from './secuencia.js';

const $ = (id) => document.getElementById(id);
const parametros = new URLSearchParams(location.search);
const DEPURAR = parametros.has('debug');
const VIDEO_PRUEBA = parametros.get('video');

const ARCHIVOS = {
  pose: { url: 'modelos/pose_landmarker_full.task', bytes: 9398198 },
  manos: { url: 'modelos/hand_landmarker.task', bytes: 7819105 },
  modelo: { url: 'modelos/lsc_tssi.onnx', bytes: 28464335 },
};
const ORDEN_CATEGORIAS = ['Cortesía', 'Números', 'Colores'];

const el = {
  escenario: $('escenario'), video: $('video'), lienzo: $('lienzo'),
  estadoTexto: $('estado-texto'), pista: $('pista'),
  subtitulo: $('subtitulo'), subPalabra: $('subtitulo-palabra'), subDetalle: $('subtitulo-detalle'),
  bienvenida: $('bienvenida'), iniciar: $('btn-iniciar'), carga: $('carga'),
  cargaAvance: $('carga-avance'), cargaTexto: $('carga-texto'), error: $('aviso-error'),
  depuracion: $('depuracion'), rendimiento: $('dato-rendimiento'),
  resultado: $('resultado'), resPalabra: $('res-palabra'), resCategoria: $('res-categoria'),
  resBarra: $('res-barra'), resConfianza: $('res-confianza'), resAlternativas: $('res-alternativas'),
  resCinta: $('res-cinta'), resMeta: $('res-meta'),
  fraseLista: $('frase-lista'), fraseVacia: $('frase-vacia'), borrar: $('btn-borrar'), voz: $('btn-voz'),
  vocabulario: $('vocabulario'), contador: $('contador'),
  esqueleto: $('btn-esqueleto'), como: $('btn-como'), dialogo: $('dialogo-como'),
};

const detector = new Detector();
const clasificador = new Clasificador();
const segmentador = new Segmentador();
const esqueleto = new Esqueleto(el.lienzo);
const seguidor = new Seguidor();
detector.reducir = !parametros.has('directo');
detector.zoom = !parametros.has('sinzoom');

let clases = [];                 // [{ id, etiqueta, categoria }]
const chipDeClase = new Map();   // índice de clase → <li> del vocabulario
const hechas = new Set();
let enCurso = 0;                 // señas esperando al clasificador
let activo = false;              // el ciclo de video está corriendo
let vozActiva = false;
let sinPersonaDesde = null;
let muyCercaDesde = null;
let deLadoDesde = null;
let sensorDeLado = false;        // el celular está acostado pero la pantalla no giró
let estadoPantalla = '';
let destelloHasta = 0;
let temporizadorPista = null;
let temporizadorSubtitulo = null;
let esqueletoVisible = true;

// Registro para las pruebas automáticas.
window.__demo = { eventos: [], registro: [], traza: [], listo: false, terminado: false, info: {} };

// ---------------------------------------------------------------- Preparación

async function descargar(archivo, alAvanzar) {
  const respuesta = await fetch(archivo.url);
  if (!respuesta.ok) throw new Error(`No se pudo descargar ${archivo.url} (${respuesta.status})`);
  if (!respuesta.body) return new Uint8Array(await respuesta.arrayBuffer());
  const lector = respuesta.body.getReader();
  const partes = [];
  let recibido = 0;
  for (;;) {
    const { done, value } = await lector.read();
    if (done) break;
    partes.push(value);
    recibido += value.length;
    alAvanzar(Math.min(recibido, archivo.bytes));
  }
  const todo = new Uint8Array(recibido);
  let posicion = 0;
  for (const parte of partes) { todo.set(parte, posicion); posicion += parte.length; }
  alAvanzar(archivo.bytes);
  return todo;
}

function mostrarCarga(fraccion, texto) {
  el.cargaAvance.style.width = `${Math.round(fraccion * 100)}%`;
  el.cargaTexto.textContent = texto;
  if (estadoPantalla === 'preparando') el.estadoTexto.textContent = texto;
}

async function preparar() {
  const datos = await (await fetch('modelos/clases.json')).json();
  clases = datos.clases;
  construirVocabulario();

  const total = Object.values(ARCHIVOS).reduce((s, a) => s + a.bytes, 0);
  const avance = { pose: 0, manos: 0, modelo: 0 };
  const informar = (clave) => (bytes) => {
    avance[clave] = bytes;
    const fraccion = (avance.pose + avance.manos + avance.modelo) / total;
    mostrarCarga(fraccion * 0.9, `Descargando modelos… ${Math.round(fraccion * 100)} %`);
  };
  const [pose, manos, modelo] = await Promise.all([
    descargar(ARCHIVOS.pose, informar('pose')),
    descargar(ARCHIVOS.manos, informar('manos')),
    descargar(ARCHIVOS.modelo, informar('modelo')),
  ]);

  mostrarCarga(0.93, 'Preparando modelos…');
  const [infoClasificador] = await Promise.all([
    clasificador.cargar(modelo.buffer),
    detector.cargar(pose, manos, { delegado: parametros.has('cpu') ? 'CPU' : 'GPU', enHilo: parametros.has('hilo') }),
  ]);

  window.__demo.info = { delegado: detector.delegado, modo: detector.modo, hilos: infoClasificador.hilos, calentamientoMs: infoClasificador.ms };
  window.__demo.listo = true;
  el.carga.classList.add('lista');
  mostrarCarga(1, 'Modelos listos');
}

// ----------------------------------------------------------------- Vocabulario

function construirVocabulario() {
  el.vocabulario.textContent = '';
  for (const categoria of ORDEN_CATEGORIAS) {
    const miembros = clases.map((c, indice) => ({ ...c, indice })).filter((c) => c.categoria === categoria);
    if (!miembros.length) continue;
    if (categoria === 'Números') {
      const ordenNumeros = ['uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'dies'];
      miembros.sort((a, b) => ordenNumeros.indexOf(a.id) - ordenNumeros.indexOf(b.id));
    } else {
      miembros.sort((a, b) => a.etiqueta.localeCompare(b.etiqueta, 'es'));
    }
    const grupo = document.createElement('section');
    grupo.className = 'grupo';
    grupo.dataset.categoria = categoria;
    const titulo = document.createElement('h3');
    titulo.append(categoria, ' ');
    const cuenta = document.createElement('small');
    cuenta.textContent = miembros.length;
    titulo.append(cuenta);
    const lista = document.createElement('ul');
    for (const m of miembros) {
      const chip = document.createElement('li');
      chip.textContent = m.etiqueta;
      chipDeClase.set(m.indice, chip);
      lista.append(chip);
    }
    grupo.append(titulo, lista);
    el.vocabulario.append(grupo);
  }
  actualizarContador();
}

function actualizarContador() {
  el.contador.innerHTML = `<b>${hechas.size}</b> / ${clases.length}`;
}

function marcarHecha(indice) {
  const chip = chipDeClase.get(indice);
  if (!chip) return;
  hechas.add(indice);
  chip.classList.add('hecha');
  chip.classList.remove('destello');
  void chip.offsetWidth;                       // reinicia la animación
  chip.classList.add('destello');
  // Si la lista tiene su propia barra de desplazamiento, se mueve solo la lista para
  // dejar a la vista la seña. La página nunca se desplaza sola.
  const caja = el.vocabulario;
  if (caja.scrollHeight > caja.clientHeight + 1) {
    const c = chip.getBoundingClientRect(), k = caja.getBoundingClientRect();
    if (c.top < k.top || c.bottom > k.bottom - 24) {
      caja.scrollTo({ top: caja.scrollTop + c.top - k.top - (k.height - c.height) / 2, behavior: 'smooth' });
    }
  }
  actualizarContador();
}

// ------------------------------------------------------------------- Pantalla

const TEXTOS_ESTADO = {
  preparando: 'Preparando modelos…',
  esperando: 'Esperando una seña',
  capturando: 'Capturando',
  analizando: 'Analizando…',
  acierto: 'Seña reconocida',
  duda: 'Seña no reconocida',
  ausente: 'Ubícate frente a la cámara',
  bloqueado: 'Baja las manos',
};

function ponerEstado(estado, texto) {
  const contenido = texto || TEXTOS_ESTADO[estado] || '';
  if (estado !== estadoPantalla) {
    estadoPantalla = estado;
    el.escenario.dataset.estado = estado;
  }
  if (el.estadoTexto.textContent !== contenido) el.estadoTexto.textContent = contenido;
}

function mostrarPista(texto, ms = 3200) {
  clearTimeout(temporizadorPista);
  if (!texto) { el.pista.hidden = true; return; }
  if (el.pista.textContent !== texto || el.pista.hidden) {
    el.pista.textContent = texto;
    el.pista.hidden = false;
  }
  if (ms) temporizadorPista = setTimeout(() => { el.pista.hidden = true; }, ms);
}

function reanimar(elemento, clase) {
  elemento.classList.remove(clase);
  void elemento.offsetWidth;
  elemento.classList.add(clase);
}

const porcentaje = (x) => `${Math.round(x * 100)} %`;

let ultimaSena = null;            // para volver a pintar la cinta si cambia el tamaño de la ventana
window.addEventListener('resize', () => {
  if (ultimaSena) pintarCinta(el.resCinta, ultimaSena.datos, ultimaSena.T);
});
animarMano($('mano'), () => !el.bienvenida.classList.contains('oculta'));

function mostrarResultado(decision, evento, salida) {
  const mejor = clases[decision.mejores[0].indice];
  const reconocida = decision.reconocida;

  // Subtítulo sobre el video
  el.subtitulo.hidden = false;
  el.subtitulo.classList.toggle('duda', !reconocida);
  el.subtitulo.classList.remove('tenue');
  el.subPalabra.textContent = reconocida ? mejor.etiqueta : 'No reconocida';
  el.subDetalle.innerHTML = '';
  if (reconocida) {
    const b = document.createElement('b');
    b.textContent = porcentaje(decision.confianza);
    el.subDetalle.append(b, ' de confianza');
  } else {
    el.subDetalle.textContent = `Lo más parecido: ${mejor.etiqueta} (${porcentaje(decision.confianza)})`;
  }
  reanimar(el.subtitulo, 'entrando');
  clearTimeout(temporizadorSubtitulo);
  temporizadorSubtitulo = setTimeout(() => el.subtitulo.classList.add('tenue'), 6000);

  // Tarjeta de resultado
  el.resultado.dataset.vacio = 'false';
  el.resultado.classList.toggle('duda', !reconocida);
  el.resultado.dataset.categoria = mejor.categoria;
  el.resPalabra.textContent = reconocida ? mejor.etiqueta : `¿${mejor.etiqueta}?`;
  el.resCategoria.textContent = mejor.categoria;
  el.resBarra.style.width = `${Math.round(decision.confianza * 100)}%`;
  el.resConfianza.textContent = porcentaje(decision.confianza);
  reanimar(el.resultado, 'entrando');

  el.resAlternativas.textContent = '';
  for (const alternativa of decision.mejores.slice(1)) {
    const fila = document.createElement('li');
    const nombre = document.createElement('span');
    nombre.textContent = clases[alternativa.indice].etiqueta;
    const mini = document.createElement('span');
    mini.className = 'mini';
    const relleno = document.createElement('i');
    relleno.style.width = `${Math.round(alternativa.confianza * 100)}%`;
    mini.append(relleno);
    const numero = document.createElement('span');
    numero.className = 'num';
    numero.textContent = porcentaje(alternativa.confianza);
    fila.append(nombre, mini, numero);
    el.resAlternativas.append(fila);
  }

  ultimaSena = evento;
  pintarCinta(el.resCinta, evento.datos, evento.T);
  el.resMeta.textContent = `${(evento.duracionMs / 1000).toFixed(1)} s · ${evento.T} cuadros · ${Math.round(salida.ms)} ms`;

  if (reconocida) {
    agregarAFrase(mejor);
    marcarHecha(decision.mejores[0].indice);
    decir(mejor.etiqueta);
  }
  destelloHasta = performance.now() + 1500;
  ponerEstado(reconocida ? 'acierto' : 'duda');
}

// ---------------------------------------------------------------------- Frase

function agregarAFrase(clase) {
  const item = document.createElement('li');
  item.textContent = clase.etiqueta;
  item.dataset.categoria = clase.categoria;
  el.fraseLista.append(item);
  el.fraseVacia.hidden = true;
  el.borrar.disabled = false;
}

el.borrar.addEventListener('click', () => {
  el.fraseLista.textContent = '';
  el.fraseVacia.hidden = false;
  el.borrar.disabled = true;
});

// ------------------------------------------------------------------------ Voz

function decir(texto) {
  if (!vozActiva || !('speechSynthesis' in window)) return;
  const frase = new SpeechSynthesisUtterance(texto.replace(/[¿?]/g, ''));
  frase.lang = 'es-CO';
  const voces = speechSynthesis.getVoices().filter((v) => v.lang && v.lang.toLowerCase().startsWith('es'));
  const voz = voces.find((v) => /co|419|mx|us/i.test(v.lang)) || voces[0];
  if (voz) frase.voice = voz;
  frase.rate = 0.95;
  speechSynthesis.cancel();
  speechSynthesis.speak(frase);
}

if (!('speechSynthesis' in window)) el.voz.hidden = true;
el.voz.addEventListener('click', () => {
  vozActiva = !vozActiva;
  el.voz.setAttribute('aria-pressed', String(vozActiva));
  if (vozActiva) decir('Voz activada'); else speechSynthesis.cancel();
});

// -------------------------------------------------------------- Clasificación

const PISTA_SIN_MANOS = 'No se alcanzaron a ver tus manos. Acércate un poco o busca más luz.';

async function clasificarEvento(evento) {
  enCurso++;
  try {
    // Antes de clasificar: ritmo de los videos de entrenamiento y huecos cortos de las manos.
    const secuencia = prepararSecuencia(evento.datos, evento.tiempos);
    const salida = await clasificador.clasificar(secuencia.datos, secuencia.T);
    const decision = decidir(salida.probabilidades, clases.length);
    const mejor = clases[decision.indice];
    if (evento.pocasManos && !creibleSinManos(mejor.id)) {
      enCurso--;
      if (DEPURAR) window.__demo.registro.push(`${(evento.tFin / 1000).toFixed(1)}s descartada:sin-manos (${mejor.id})`);
      mostrarPista(PISTA_SIN_MANOS, 4500);
      return;
    }
    window.__demo.eventos.push({
      id: mejor.id, etiqueta: mejor.etiqueta, confianza: decision.confianza, reconocida: decision.reconocida,
      inicio: evento.tInicio, fin: evento.tFin, cuadros: evento.T, cuadrosModelo: secuencia.T,
      rellenados: secuencia.rellenados, ms: salida.ms,
      mejores: decision.mejores.map((m) => [clases[m.indice].etiqueta, m.confianza]),
    });
    enCurso--;
    mostrarResultado(decision, evento, salida);
  } catch (error) {
    enCurso--;
    console.error(error);
    mostrarPista('No se pudo analizar esa seña. Inténtalo de nuevo.');
  }
}

// ------------------------------------------------------------- Ciclo de video
//
// El video de la cámara se muestra directo, tal como llega. Aparte:
//   1. Cuando MediaPipe está libre, analiza el cuadro más reciente.
//   2. El esqueleto se dibuja encima del video en cada pintada de la pantalla; el
//      Seguidor (esqueleto.js) lo mueve con suavidad entre un análisis y el siguiente.

let analizando = false;
let latencia = 0;                  // ms entre capturar un cuadro y tener sus puntos
let intervalo = 0;                 // ms entre dos análisis seguidos
let tAnalisisPrevio = 0;
let tiempoVideoPrevio = -1;
let hayPuntos = false;
let tPintado = 0;
let pintadoPendiente = false;
const suavizar = (previo, valor, peso = 0.15) => (previo ? previo + (valor - previo) * peso : valor);

function programar() {
  if (!activo) return;
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    el.video.requestVideoFrameCallback(alCuadro);
  } else {
    requestAnimationFrame((ahora) => alCuadro(ahora, null));
  }
}

function alCuadro(_, meta) {
  if (!activo) return;
  programar();
  const video = el.video;
  if (analizando || video.readyState < 2 || !video.videoWidth) return;
  if (!meta) {                                    // sin requestVideoFrameCallback: evitar cuadros repetidos
    if (video.currentTime === tiempoVideoPrevio) return;
    tiempoVideoPrevio = video.currentTime;
  }
  const reloj = performance.now();
  // Con un video de prueba, el tiempo de las señas es el del propio video.
  const tSena = VIDEO_PRUEBA ? (meta ? meta.mediaTime : video.currentTime) * 1000 : reloj;

  analizando = true;
  analizar(video, reloj, tSena).catch((error) => console.error(error)).finally(() => { analizando = false; });
}

async function analizar(video, reloj, tSena) {
  const ancho = video.videoWidth, alto = video.videoHeight;
  detector.cajaFija = segmentador.estado === 'capturando';
  const resultado = await detector.detectar(video, reloj, (pose) => {
    seguidor.ponerCuerpo(pose, reloj);           // el cuerpo se mueve sin esperar a las manos
    hayPuntos = true;
    pedirPintado();
  });
  const llegada = performance.now();
  latencia = suavizar(latencia, llegada - reloj);
  if (tAnalisisPrevio) intervalo = suavizar(intervalo, reloj - tAnalisisPrevio);
  tAnalisisPrevio = reloj;

  // El modelo y el segmentador trabajan en la proporción de los videos de entrenamiento;
  // el dibujo usa los puntos tal como salen, sobre el video real.
  const factor = factorDeProporcion(ancho, alto);
  segmentador.factorX = factor;
  const evento = segmentador.procesar(corregirProporcion(resultado.vector, factor), tSena);
  if (DEPURAR) {
    const m = segmentador.medida, c = detector.caja, n = (x) => (Number.isFinite(x) ? +x.toFixed(2) : null);
    const mano = (desde) => resultado.vector.subarray(desde, desde + 63).some((x) => x !== 0);
    window.__demo.traza.push([Math.round(tSena), segmentador.estado[0], n(m.alturas[0]), n(m.alturas[1]),
      n(segmentador.base[0]), n(segmentador.base[1]), n(m.borde), mano(99) ? 1 : 0, mano(162) ? 1 : 0, n(c.w * c.h),
      resultado.pose ? n(resultado.pose[15 * 4 + 3]) : null, resultado.pose ? n(resultado.pose[16 * 4 + 3]) : null]);
  }
  if (evento) {
    if (DEPURAR) window.__demo.registro.push(`${(tSena / 1000).toFixed(1)}s ${evento.tipo}${evento.motivo ? ':' + evento.motivo : ''}`);
    if (evento.tipo === 'sena') {
      clasificarEvento(evento);
    } else if (evento.tipo === 'descartada' && evento.motivo === 'larga') {
      mostrarPista('Baja las manos para terminar la seña', 4000);
    } else if (evento.tipo === 'descartada' && evento.motivo === 'sin-manos') {
      mostrarPista(PISTA_SIN_MANOS, 4500);
    }
  }

  if (detector.modo === 'hilo') seguidor.ponerCuerpo(resultado.pose, reloj);
  seguidor.ponerManos(resultado.vector, reloj);
  hayPuntos = true;
  revisarOrientacion(resultado.pose, ancho, alto, llegada);
  pedirPintado();
  actualizarEstado(llegada);
  if (DEPURAR) depurar();
}

function pedirPintado() {
  if (pintadoPendiente) return;
  pintadoPendiente = true;
  requestAnimationFrame(pintar);
}

function pintar(ahora) {
  pintadoPendiente = false;
  if (!activo) return;
  if (!esqueletoVisible || !hayPuntos) {
    esqueleto.limpiar();
    tPintado = 0;
    return;
  }
  const dt = tPintado ? Math.min(100, ahora - tPintado) : 16;
  tPintado = ahora;
  esqueleto.dibujar(seguidor.avanzar(performance.now(), dt), el.video.videoWidth, el.video.videoHeight,
    segmentador.estado === 'capturando');
  // Se pinta de seguido mientras estén llegando análisis.
  if (performance.now() - tAnalisisPrevio < 1000 || analizando) pedirPintado(); else tPintado = 0;
}

/** Avisa si la imagen llega acostada (celular de lado con la rotación bloqueada). */
function revisarOrientacion(pose, ancho, alto, ahora) {
  let deLado = sensorDeLado;
  if (pose) {
    const dx = (pose[11 * 4] - pose[12 * 4]) * ancho, dy = (pose[11 * 4 + 1] - pose[12 * 4 + 1]) * alto;
    const fiable = pose[11 * 4 + 3] > 0.5 && pose[12 * 4 + 3] > 0.5;
    if (fiable) deLado = Math.abs(dy) > 1.6 * Math.abs(dx);
  }
  if (!deLado) { deLadoDesde = null; return; }
  if (deLadoDesde === null) deLadoDesde = ahora;
  if (ahora - deLadoDesde > 1500) {
    mostrarPista('La imagen llega de lado. Activa el giro automático del celular o úsalo en vertical.', 4000);
    deLadoDesde = ahora;
  }
}

window.addEventListener('devicemotion', (evento) => {
  const g = evento.accelerationIncludingGravity;
  if (!g || g.x === null) return;
  const pantallaVertical = window.innerHeight > window.innerWidth;
  sensorDeLado = pantallaVertical && Math.abs(g.x) > 7.5 && Math.abs(g.y) < 4;
});

function actualizarEstado(ahora) {
  const m = segmentador.medida;

  // Avisos de encuadre (solo mientras no se está haciendo una seña).
  if (!m.persona) { if (sinPersonaDesde === null) sinPersonaDesde = ahora; } else sinPersonaDesde = null;
  // Hombros que ocupan casi la mitad del ancho del cuadro: los brazos no van a caber.
  const hombros = m.persona ? m.ancho / segmentador.factorX : 0;
  if (hombros > 0.44) { if (muyCercaDesde === null) muyCercaDesde = ahora; } else muyCercaDesde = null;

  if (segmentador.estado === 'capturando') {
    ponerEstado('capturando', `Capturando · ${(segmentador.duracionActual / 1000).toFixed(1)} s`);
    return;
  }
  if (segmentador.estado === 'bloqueado') { ponerEstado('bloqueado'); return; }
  if (enCurso > 0) { ponerEstado('analizando'); return; }
  if (ahora < destelloHasta) return;            // deja ver un momento el resultado
  if (sinPersonaDesde !== null && ahora - sinPersonaDesde > 1200) {
    ponerEstado('ausente');
    return;
  }
  ponerEstado('esperando');
  if (muyCercaDesde !== null && ahora - muyCercaDesde > 2500) {
    mostrarPista('Aléjate un poco para que quepan tus brazos', 2500);
    muyCercaDesde = ahora;
  }
}

let ultimoRendimiento = 0;
function depurar() {
  const m = segmentador.medida;
  const n = (x) => (Number.isFinite(x) ? x.toFixed(2) : '—');
  const porSegundo = intervalo ? 1000 / intervalo : 0;
  el.depuracion.textContent =
    `${porSegundo.toFixed(0)} análisis/s · latencia ${latencia.toFixed(0)} ms\n` +
    `MediaPipe ${detector.delegado} (${detector.modo}) · clasificador ${clasificador.hilos} hilo(s)\n` +
    `estado ${segmentador.estado} · persona ${m.persona ? 'sí' : 'no'} · hombros ${n(m.ancho)}\n` +
    `altura izq ${n(m.alturas[0])} · der ${n(m.alturas[1])} · reposo ${n(segmentador.base[0])} / ${n(segmentador.base[1])}\n` +
    `video ${el.video.videoWidth}×${el.video.videoHeight} · factor ${segmentador.factorX.toFixed(2)} · ` +
    `zoom ${(1 / Math.sqrt(detector.caja.w * detector.caja.h)).toFixed(1)}×`;
  const ahora = performance.now();
  if (ahora - ultimoRendimiento > 500) {
    ultimoRendimiento = ahora;
    el.rendimiento.textContent = `${porSegundo.toFixed(0)} análisis/s`;
  }
}

// --------------------------------------------------------------------- Inicio

function explicarErrorCamara(error) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return 'Este navegador no permite usar la cámara aquí. Abre la página con https o desde localhost.';
  }
  switch (error && error.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Falta el permiso para usar la cámara. Actívalo en el candado de la barra de direcciones y vuelve a intentarlo.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No se encontró ninguna cámara en este dispositivo.';
    case 'NotReadableError':
      return 'Otra aplicación está usando la cámara. Ciérrala y vuelve a intentarlo.';
    default:
      return 'No se pudo abrir la cámara. Vuelve a intentarlo.';
  }
}

function mostrarError(texto) {
  el.error.textContent = texto;
  el.error.hidden = false;
}

async function abrirFuente() {
  if (VIDEO_PRUEBA) {
    el.video.src = VIDEO_PRUEBA;
    el.video.loop = false;
    el.video.addEventListener('ended', async () => {
      // Deja terminar lo que esté en curso antes de avisar a la prueba.
      while (enCurso > 0) await new Promise((r) => setTimeout(r, 50));
      window.__demo.terminado = true;
    });
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('sin-camara');
  const flujo = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
  });
  el.video.srcObject = flujo;
}

/** Acomoda el escenario a la forma del video (cambia cuando el celular gira). */
function ajustarAlVideo() {
  const ancho = el.video.videoWidth, alto = el.video.videoHeight;
  if (!ancho || !alto) return;
  el.escenario.style.setProperty('--ar', `${ancho} / ${alto}`);
  segmentador.reiniciar();
  seguidor.reiniciar();
  requestAnimationFrame(() => { esqueleto.ajustar(); pedirPintado(); });
}

async function iniciar() {
  el.error.hidden = true;
  el.iniciar.disabled = true;
  try {
    await abrirFuente();
  } catch (error) {
    el.iniciar.disabled = false;
    mostrarError(explicarErrorCamara(error));
    return;
  }

  // En el modo de prueba el video no arranca hasta que los modelos estén listos.
  if (VIDEO_PRUEBA) {
    await new Promise((resolver) => {
      if (el.video.readyState >= 1) resolver(); else el.video.addEventListener('loadedmetadata', resolver, { once: true });
    });
  } else {
    await el.video.play().catch(() => {});
    if (!el.video.videoWidth) {
      await new Promise((resolver) => el.video.addEventListener('loadedmetadata', resolver, { once: true }));
    }
  }

  ajustarAlVideo();
  el.video.addEventListener('resize', ajustarAlVideo);      // el celular giró: cambia el tamaño del video
  el.escenario.classList.add('con-video');
  el.bienvenida.classList.add('oculta');
  ponerEstado('preparando', el.cargaTexto.textContent);
  esqueleto.ajustar();

  try {
    await preparacion;
  } catch (error) {
    console.error(error);
    el.bienvenida.classList.remove('oculta');
    mostrarError('No se pudieron cargar los modelos. Revisa tu conexión y recarga la página.');
    return;
  }

  if (VIDEO_PRUEBA) await el.video.play().catch((e) => console.error(e));
  activo = true;
  ponerEstado('esperando');
  if (DEPURAR) { el.depuracion.hidden = false; el.rendimiento.hidden = false; }
  programar();
}

el.iniciar.addEventListener('click', iniciar);

el.esqueleto.addEventListener('click', () => {
  esqueletoVisible = !esqueletoVisible;
  tPintado = 0;
  el.esqueleto.setAttribute('aria-pressed', String(esqueletoVisible));
  pedirPintado();
});

el.como.addEventListener('click', () => el.dialogo.showModal());
el.dialogo.addEventListener('click', (evento) => {
  if (evento.target === el.dialogo) el.dialogo.close();     // clic fuera del contenido
});

new ResizeObserver(() => { esqueleto.ajustar(); if (activo) pedirPintado(); }).observe(el.escenario);
document.querySelector('.medidor-umbral').style.left = `${UMBRAL_CONFIANZA * 100}%`;

const preparacion = preparar();
preparacion.catch((error) => {
  console.error(error);
  mostrarCarga(0, 'No se pudieron cargar los modelos');
  mostrarError('No se pudieron cargar los modelos. Revisa tu conexión y recarga la página.');
});

if (VIDEO_PRUEBA) iniciar();
