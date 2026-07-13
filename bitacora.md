# Bitácora

## 2026-07-13

### 1. Tercera tarea generalizada: jpazm/hw1 ("Tarea 3" — demanda de café, conducta, contrafactuales)

Construcción de `jpazm/hw1/solutions/rubric.yaml` desde cero (9 ítems: P2a-c demanda inversa
lineal/log-lineal/elasticidad, P3a-b costo marginal Cournot/Stackelberg, P4a-d cuatro
contrafactuales), siguiendo el patrón ya establecido en `pbordon/hw2` y `jpazm/hw2` — pesos sumando
1.0, `puntos_teoricos` por ítem, `parejas_permitidas: true`, `salida` apuntando a `notas.xlsx` hoja
`ii. tarea_1`. El profesor ya había entregado una pauta completa (`coffee_report.pdf`), por lo que no
hizo falta reproducir un informe LaTeX nuevo (Opción A). `notas_generales` documenta varias
excepciones legítimas: κ=0.5 comunicado aparte del enunciado impreso, la regla de "apagado" para
cantidades negativas, instrumento débil en spec 2, contradicción entre los dos documentos de la pauta
sobre elástica/inelástica, y que el índice de Lerner no es exigible (el enunciado solo pide figuras
de markup).

### 2. Hallazgo central de la sesión: falla sistemática de subcaptura en el pipeline extractor→verificador

Corridas las 12 entregas (17 alumnos) por el pipeline estándar (`extractor-resultados` →
`verificador-explicacion`), aparecieron varias notas bajas y un supuesto "patrón de clase" (casi toda
la clase omitiendo el mismo punto teórico en dos ítems de contrafactuales). Antes de cerrar, y por
precaución explícita del TA ante posibles reclamos ("no vaya a ser que su entrega estaba en el
informe y no fue detectado"), se releyeron directamente — con `Read`, el documento completo, sin
sub-agentes — 3 de las notas más bajas. **2 de esas 3 tenían contenido real, sustancial, que el
pipeline había marcado como ausente.** Dado ese resultado (no un caso aislado), se escaló a releer
las **12 entregas completas**: el resultado fue que **9 de 12 se recalificaron al alza**, incluyendo
una nota media-alta (6.0/7.0) — es decir, el problema no estaba correlacionado con la nota, afectaba
transversalmente a la mayoría de la clase. El supuesto "patrón de clase" resultó ser, en su mayor
parte, un artefacto: el punto que se creía omitido por casi todos estaba de hecho explícito en 9 de
12 informes.

**Causa raíz:** el diseño de la skill ya decía "pásale a `verificador-explicacion` el texto crudo del
extractor, no lo resumas" — pero en la práctica, al orquestar 12 dispatches en paralelo, se terminó
construyendo una versión condensada del texto de cada alumno antes de pasarla al verificador, y esa
condensación perdió pasajes reales. Un caso ilustrativo: un informe decía literalmente *"la condición
de recuperación para los seguidores colapsa a la misma ecuación que en Cournot"* — la idea central de
uno de los puntos supuestamente "omitidos por toda la clase" — pero esa frase no sobrevivió al
resumen que se le pasó al verificador.

**Un caso adicional de baja real** (no atribuible a subcaptura): un alumno cuya explicación de un
contrafactual solo afirmaba que el valor era "positivo" sin abordar en ningún momento que el efecto
agregado de mercado es pequeño — la relectura confirmó que el puntaje original estaba, si acaso,
sobrevalorado, y se corrigió a la baja. Sirve como contraste: no toda relectura termina en una subida.

### 3. Cambios institucionalizados para que esto no se repita en futuras correcciones

- **`.claude/agents/extractor-resultados.md`**: (a) instrucción explícita de no condensar la cita a
  "la oración más representativa" — transcribir el pasaje completo aunque sea largo; (b) alerta sobre
  argumentos acumulativos: un mecanismo establecido en una sección puede invocarse por referencia en
  una sección posterior sin repetirse ("como vimos anteriormente...") — el extractor debe rastrear y
  transcribir también el pasaje original, no descartar la referencia por no estar repetida palabra por
  palabra donde nominalmente se responde la pregunta.
- **`.claude/agents/verificador-explicacion.md`**: nuevo caso degenerado — si el texto recibido es
  sospechosamente breve para la cantidad de `puntos_teoricos` que exige el ítem, márcalo explícitamente
  como observación (no lo asumas en ningún sentido), para que quien despache el agente decida si vale
  verificar contra el documento original.
- **`.claude/skills/corregir-tarea/SKILL.md`**: (a) instrucción explícita de pasar el
  `texto_explicación` completo del extractor a `verificador-explicacion`, nunca una versión curada por
  quien orquesta; (b) **nuevo paso obligatorio de calibración** (paso 5): antes de cerrar cualquier
  nota, releer directamente (sin agentes) 2-3 entregas completas, **cruzando la distribución de notas**
  (no solo las más bajas — el hallazgo de esta sesión fue justamente que una nota media-alta también
  tenía el problema). Si la muestra encuentra una sola discrepancia real, escalar a releer todas las
  entregas antes de cerrar — no tratarla como aislada; (c) regla de "patrón de clase": nunca decidir
  unilateralmente si se excusa o se penaliza una omisión compartida por la clase — presentarle al TA
  la evidencia concreta ítem por ítem, y solo tras confirmar (con texto completo, no resúmenes) que el
  patrón es real. Tope de 1 bono de patrón de clase por alumno por tarea si el TA aprueba excusarlo.

### 4. Resultado y entregables

Notas finales (`ii. tarea_1`, 17 alumnos): rango 4.5–7.0. Dos casos límite documentados con especial
cuidado: una pareja que queda en nota máxima exacta (7.0) — confirmada tras **dos** relecturas
directas independientes (una general, una buscando activamente fallas) y cruce numérico contra 3
entregas independientes, sin encontrar ningún error; y un alumno con la única corrección a la baja de
la sesión. Los 12 `feedback.md` (dentro de `submissions/`, no versionados) citan pasaje textual del
informe por cada ítem penalizado, para que el TA pueda responder cualquier correo con la evidencia
exacta a mano.

### Pendiente / próximos pasos

- Notas de `jpazm/hw1` quedan **propuestas**, pendientes de que el TA las cierre (misma convención que
  `jpazm/hw2`).
- El paso de calibración obligatoria (nuevo paso 5 de la skill) no se aplicó retroactivamente a
  `jpazm/hw2` ni a `pbordon/hw2` — quedó fuera de alcance de esta sesión. Si el TA quiere la misma
  garantía sobre esas correcciones ya cerradas, habría que releer una muestra de esas entregas también.
- Vale la pena, en la próxima tarea que se corrija, confirmar en la práctica que el nuevo paso de
  calibración efectivamente se ejecuta antes de escribir `feedback.md` (no solo que quedó escrito en
  la skill) — es fácil que un paso "antes de cerrar notas" se salte bajo presión de terminar rápido.

## 2026-07-11

### 1. Nueva arquitectura de corrección: rúbrica como dato + agentes

Problema de fondo detectado: la clave de corrección de `pbordon/hw2` (18 ítems, valores esperados,
SE, tolerancias) estaba escrita a mano dentro de la skill `corregir-tarea`, mezclada con el
procedimiento genérico. No escalaba a tareas nuevas sin editar la skill cada vez.

Rediseño (inspirado parcialmente en `pedrohcgs/claude-code-my-workflow`, cherry-pickeando solo dos
patrones — sin importar todo su aparataje de gates/hooks, que es de más de lo que este repo
necesita):

- **`rubric.yaml`** por tarea (`solution(s)/rubric.yaml`): ítems, valores esperados/tolerancias,
  `puntos_teoricos` mínimos por ítem, pesos. Lo genera `pauta-tarea`; lo consume `corregir-tarea`.
  Nunca más una rúbrica hardcodeada dentro de una skill.
- **Agente `extractor-resultados`** (`.claude/agents/`): solo lectura, recupera valor numérico +
  texto de explicación de una entrega. Maneja Stata/R/Python indistintamente.
- **Agente `verificador-explicacion`** (`.claude/agents/`): solo lectura, **contexto fresco**
  (patrón Chain-of-Verification) — evalúa si la explicación cubre los puntos teóricos SIN ver el
  resultado numérico ni el puntaje, para no heredar el sesgo de "el número dio bien, seguro explicó
  bien".
- **Scorer determinístico**: la aritmética de tolerancias (SE-anchored, % relativo, signo) se
  calcula con código, no con juicio de un LLM.
- Principio de diseño que quedó explícito para futuras skills/agentes de este repo: **skill =
  orquesta + checkpoints humanos** (solo el hilo principal puede pausar a preguntar); **agente =
  aislamiento** (paralelismo, evitar ensuciar contexto, o ceguera deliberada); **script = aritmética
  pura**, nunca un LLM.

Entregable de validación: `pbordon/hw2/solution/rubric.yaml` (migración de la tabla de 18 ítems que
antes vivía en la skill).

### 2. Primera generalización real: corrección de jpazm/hw2 (Homework 4, aerolíneas)

Segunda tarea del curso — Parte II (jpazm), estructura de carpetas distinta a pbordon
(`assignment/solutions/submissions` en vez de `solution/submissions`), con parejas permitidas y
salida a un libro de notas consolidado (`docencia-web/notas.xlsx`) en vez de un archivo nuevo.

Cambios quirúrgicos a `corregir-tarea` para soportar esto de forma genérica (no específica de esta
tarea): manejo de parejas (una entrega → dos códigos de alumno, mismos puntajes) y salida
configurable (`rubric.yaml.salida` apunta a un libro/hoja existente, replicando el estilo de una
hoja hermana en vez de asumir el formato estándar).

`jpazm/hw2/solutions/rubric.yaml`: 11 ítems (P1a-c, P2a-e, P3a-c), pesos 20/45/35% por parte. Nota
de rúbrica relevante: el PDF del enunciado marca la Parte 3 como "(optional)", pero el profesor
indicó directamente que las 3 partes son obligatorias — se siguió esa instrucción sin matices.

Corrección de las 12 entregas (17 alumnos, 5 parejas + 7 individuales), roster confirmado contra
`2026_fall/grades.xlsx`. Resultado: notas entre 5.3 y 6.9. Como es una tarea "run-and-interpret"
(scripts ya dados), la Parte 1 salió perfecta en las 12 entregas — el pipeline es determinístico, no
hay diferenciación real ahí. Toda la dispersión de nota vino de la interpretación (Parte 2) y de la
Parte 3.

Hallazgo transversal para que el TA lo tenga presente (no es un caso aislado): casi toda la clase
cubrió débilmente dos puntos teóricos específicos — que D̄_AB no es un criterio de selección de
rutas (Parte 2b) y que el repricing es exactamente cero en κ=0 (Parte 3c). Y 4 de 12 entregas no
reportaron en el cuerpo del informe la tabla/conteo de vuelos "at-risk" por κ que pide la Parte 3b,
aunque el dato existe en sus propios CSV.

Output: hoja nueva `ii. tarea_2` en `docencia-web/notas.xlsx` (mismo estilo que sus hojas hermanas,
columna `ID` sin `Nombre` aparte) + `feedback.md` por entrega + respaldo de extracción/verificación
en `jpazm/hw2/_correccion/`. **Notas propuestas, no cerradas** — quedan pendientes de revisión del
TA antes de considerarse definitivas.

### 3. Mantenimiento del repo

- `_correccion/` de jpazm se movió de dentro de `submissions/` a ser hermana de `submissions/`,
  igual que el `_grading/` de pbordon (consistencia: separar trabajo de corrección del TA de las
  entregas crudas de los alumnos).
- `.gitignore`: se generalizó `submissions/` a un patrón que cubre cualquier
  profesor/tarea (`2026_fall/01-homeworks/*/hw*/submissions/`) — antes solo excluía
  `pbordon/hw1/submissions` explícitamente. Se agregaron `.DS_Store`, `__MACOSX/`, `*.obj`,
  `.ipynb_checkpoints/`, y el patrón de `_grading/`/`_correccion/` generalizado. Se destrackearon 8
  `.DS_Store` que ya estaban versionados por error (quedan en disco, solo se sacaron del índice de
  git).
- `CLAUDE.md`: se agregó la estructura de `jpazm/` (assignment/solutions/submissions) y de
  `docencia-web/` (notas consolidadas, convención de hojas `i.`/`ii.`) al árbol del repo; se
  actualizó la sección de `rubric.yaml` para mencionar los campos `parejas_permitidas` y `salida`, y
  la lista de exclusiones de `.gitignore`.

### Pendiente / próximos pasos

- TA revisa las notas de jpazm/hw2 propuestas antes de cerrarlas (ver casos límite detallados en el
  resumen de la sesión de corrección).
- No se backfilleó `rubric.yaml` para `pbordon/hw1` (ya corregida, no se pidió) — trivial de hacer
  cuando haga falta, siguiendo la plantilla de `pbordon/hw2`.
- Queda pendiente decidir si `feedback-correcciones.md` e `instrucciones.md` (sueltos en la raíz del
  repo, con problemas de encoding) se archivan, se limpian o se dejan — no se tocaron en esta
  sesión.
