# Bitácora

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
