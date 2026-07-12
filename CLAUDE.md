# CLAUDE.md

Guía operativa del repositorio. Léela antes de preparar pautas o corregir entregas.

## ¿Qué es esto?

Material de apoyo del curso **Métodos Avanzados para la Organización Industrial Empírica**, electivo del Magíster en Economía de la Universidad de Chile (profesores: Paola Bordón y José Manuel Paz y Miño).

Contiene clases, controles de lectura (*quizzes*) y tareas (*homeworks*), con sus respaldos, soluciones y archivos de apoyo.

**Mi rol es de TA.** Mis responsabilidades sobre este repo son:
1. Mantener el orden y la gestión documental.
2. **Preparar las pautas** de tareas y quizzes a partir del código de solución del profesor.
3. **Corregir las entregas** de los estudiantes contra esas pautas.

## Estructura del repositorio

```
empirical_oi/
├── archive/                 # Material antiguo por ordenar. NO es la fuente de verdad del semestre.
│   ├── hw2/, hw3/           # Tareas de versiones previas (código + informe + datos).
│   └── demand_estimation/   # Ejemplos de BLP en Python (pyblp, datos de Nevo). Útil de referencia.
│
└── 2026_fall/               # TODO el material vigente (Otoño 2026). Fuente de verdad.
    ├── 01-homeworks/pbordon/ # Tareas organizadas por profesor.
    │   ├── hw1/             # Tarea 1: Estimación de funciones de producción (OP, LP, ACF).
    │   │   ├── solution/    # ← PLANTILLA DE REFERENCIA de una pauta (ver "Anatomía de una pauta").
    │   │   └── submissions/ # Entregas de estudiantes (carpetas/zips, formato heterogéneo).
    │   └── hw2/             # Tarea 2: Estimación de demanda (Logit, Nested Logit, BLP). Datos OTC.
    │       ├── raw/         # Datos crudos (OTC_Data, OTC_Demographics, OTC_Instruments, v).
    │       ├── solution/    # code/hw2_logit_nl.py y hw2_blp.py (Python) tienen la solución reproducida;
    │       │                # report/hw2-report.tex + rubric.yaml son la pauta de corrección vigente.
    │       │                # hw2-solution.pdf es la pauta del profesor (respaldo de lectura económica).
    │       └── submissions/ # Entregas (zips/PDF).
    │   └── jpazm/           # Tareas de la Parte II del curso (jpazm). Layout distinto a pbordon:
    │       └── hwN/
    │           ├── assignment/  # Enunciado (PDF) + datos + scripts que se le dan al alumno.
    │           ├── solutions/   # (plural) Pauta del profesor/RA (PDF) + rubric.yaml destilado de ahí.
    │           └── submissions/ # Entregas — a veces en pareja (dos nombres en la carpeta).
    ├── 02-quizzes/pbordon/  # Controles de lectura.
    │   ├── qz1/             # Levinsohn & Petrin. Incluye qz1-answers.pdf y qz1-grades.xlsx (formato de notas).
    │   ├── qz2/             # Nevo (EMA 2001).
    │   └── qz3/             # Genesove (1998).
    ├── 03-lectures/         # Diapositivas de clase (pbordon/, jpazm/).
    └── docencia-web/        # Libro de notas consolidado: notas.xlsx, con una hoja por
                              # lectura/tarea ("i. tarea_N" = Parte I/pbordon, "ii. tarea_N" =
                              # Parte II/jpazm), columnas ID + ítems + Total % + Nota.
```

**Relevancia de la información:**
- `2026_fall/` es lo único vigente; `archive/` solo se consulta como referencia.
- Dentro de cada tarea, `solution/` es la **fuente de la verdad** (resultados correctos). `submissions/` es lo que se corrige contra ella.
- El enunciado oficial de cada tarea es el PDF en la carpeta de la tarea (p. ej. `hw2/hw-2.pdf`).

## Anatomía de una pauta (referencia: `hw1/solution/`)

Cada tarea resuelta sigue esta estructura. **Replicar este layout en cada tarea nueva.**

```
solution/
├── src/          # Datos originales de entrada (.dta). p. ej. ChileAnalysis.dta
├── code/         # Código del profesor con todas las respuestas. p. ej. hw1-solution.do
├── bld/          # Outputs generados (build). NO se editan a mano.
│   ├── tab/      # Tablas en LaTeX (.tex), incluidas con \input
│   └── fig/      # Figuras (.pdf), incluidas con \includegraphics
├── report/       # El informe-pauta.
│   ├── hwX-report.tex   # Enunciado + respuestas. ESTE es el entregable autocontenido.
│   └── build/          # PDF compilado y auxiliares.
├── rubric.yaml   # Contraparte estructurada del informe: ítems, valores/tolerancias, puntos
│                 # teóricos mínimos y pesos. La consume la skill `corregir-tarea`.
└── out/          # (opcional) outputs finales.
```

### El informe-pauta (`report/hwX-report.tex`)

Es un **reporte autocontenido**: el lector debe entender pregunta y respuesta sin abrir nada más. Convenciones:

- Clase `\documentclass[answers]{exam}`, en español (`babel`, `inputenc utf8`), con `booktabs`.
- Cada pregunta es un `\question`; la respuesta va dentro de `\begin{solution} ... \end{solution}`.
- **El enunciado ya viene copiado** en el `.tex`. Solo se rellena el contenido de cada `solution`.
- Cada `solution` contiene, en este orden:
  1. **Montaje** (1-2 líneas): qué se estima, con qué método, qué variable es proxy/estado/libre.
  2. **Resultado**: tabla `\input{../bld/tab/nombre.tex}` y/o figura `\includegraphics{../bld/fig/nombre.pdf}` (rutas **relativas a `report/`**, por eso el `../bld/`).
  3. **Interpretación económica** (2-4 líneas): el hallazgo y su lectura económica (signos, magnitudes, comparaciones entre modelos, retornos a escala, sesgos), no solo describir números.
  4. **Trazabilidad**: `Ver código: \texttt{code/hwX-solution.do} líneas N--M.`
- Para figuras que pueden no existir aún, usar el patrón `\IfFileExists{...}{...}{\fbox{...placeholder...}}` (ver `hw1-report.tex` líneas 197-219) para que el `.tex` compile igual.

### El rubric.yaml (`solution/rubric.yaml`)

Es la **contraparte estructurada** del informe-pauta: mismos ítems, pero en formato que un agente puede leer sin reinterpretar prosa. Por cada ítem trae `resultado` (valor(es) esperado(s), SE si aplica, tipo de tolerancia), `puntos_teoricos` (2-4 ideas mínimas que la explicación de un alumno debe cubrir, destiladas de la interpretación económica del informe) y `peso`. Lo genera `pauta-tarea` (o se destila directo de una pauta PDF ya entregada por el profesor, si no hace falta construir un informe LaTeX nuevo) y lo consume `corregir-tarea` — nunca se hardcodea la clave de corrección de una tarea dentro de una skill. Campos opcionales: `parejas_permitidas` (si la tarea admite grupos de dos) y `salida` (si la nota va a un libro/hoja distinto del `hwN_grades.xlsx` estándar — p. ej. una hoja nueva en `docencia-web/notas.xlsx`). Ver `pbordon/hw2/solution/rubric.yaml` (caso simple) y `jpazm/hw2/solutions/rubric.yaml` (con parejas y salida a `notas.xlsx`) como plantillas.

### Las tablas (`bld/tab/*.tex`)

- Fragmentos LaTeX autocontenidos: `\begin{table}[H] ... \begin{tabular} ... \end{table}`, con `\caption`, `booktabs` (`\toprule/\midrule/\bottomrule`) y una nota al pie explicando proxy/variable dependiente.
- Coeficientes redondeados (3 decimales en hw1).

### El código de solución (`code/hwX-solution.do`)

- Stata (`.do`). Organizado con **marcadores de sección por pregunta** (`/*** Pregunta N ***/`).
- Los coeficientes esperados van como **comentarios en línea** (p. ej. `/*0.6930964*/`). Esa es la fuente de los números correctos.
- **Gotcha de rutas:** el `.do` original abre los datos con una ruta absoluta de Windows (`use "C:\Users\Paola Bordon\Dropbox\..."`). Para reproducir hay que apuntar a `src/` (hw1) o `raw/` (hw2).

## Flujo 1 — Preparar la pauta de una tarea

1. **Leer el enunciado** (PDF de la tarea) para saber qué pide cada pregunta.
2. **Leer el código de solución** (`code/*.do`) e identificar qué resultado produce cada sección y cuáles son los coeficientes/valores correctos (comentarios en línea o reejecutando).
3. **Generar los outputs** en `bld/tab/` y `bld/fig/` con los números correctos.
4. **Rellenar** cada `\begin{solution}` en `report/hwX-report.tex` con montaje + resultado + interpretación + trazabilidad.
5. **Generar `rubric.yaml`** (contraparte estructurada del informe: ítems, resultado/tolerancias, puntos_teoricos, pesos) — ver la sección "El rubric.yaml" arriba.
6. El producto final es un informe **autocontenido** (respuesta modelo) + `rubric.yaml` (pauta de corrección que consume `corregir-tarea`).

## Flujo 2 — Corregir entregas (`submissions/`)

Se corrige cada entrega **contra `rubric.yaml`**, evaluando dos ejes independientes: (a) **resultado numérico** (¿coincide con el esperado, dentro de la tolerancia declarada?) y (b) **cobertura de los puntos teóricos** que la pauta exige en la explicación. La skill `corregir-tarea` implementa esto despachando dos agentes de solo lectura por entrega (`extractor-resultados` para (a), `verificador-explicacion` para (b), este último en contexto fresco para no heredar el sesgo del resultado numérico) más un scorer determinístico para la aritmética de tolerancias — ver `.claude/agents/` y `.claude/skills/corregir-tarea/SKILL.md` para el detalle.

**Realidades de las entregas (esperar desorden):**
- Vienen como `.zip` (a veces PDF suelto), con nombres heterogéneos. Hay copias `__MACOSX/` y `.DS_Store` que se ignoran.
- Software mixto: Stata (`.do`), Python (`.ipynb`, `.venv`), R. No asumir uno solo.
- Algunas carpetas usan ID de estudiante (`2026830496-20260`), otras el nombre.

**Recomendación de proceso:** corregir ítem por ítem usando `rubric.yaml` como referencia; reportar por entrega el resultado obtenido vs. esperado y un comentario sobre el razonamiento. Mantener el formato de notas consistente con `02-quizzes/pbordon/qz1/qz1-grades.xlsx`.

## Convenciones generales y gotchas

- **Idioma:** todo en español (informes, comentarios, tablas). UTF-8.
- **Rutas de datos:** nunca confiar en las rutas absolutas de los `.do`; apuntar a `src/`/`raw/` local.
- **`.gitignore`** ya excluye `archive/`, **todas** las carpetas `submissions/` (cualquier profesor/tarea — nunca se versionan: contienen PII y datos pesados de los alumnos), las carpetas de trabajo de corrección (`_grading/`, `_correccion/`), y archivos de datos (`*.csv`, `*.dta`, `*.xlsx`, `*.zip`, `*.log`, `*.obj`). Los datos pesados y las entregas no se versionan.
- **Ruido de macOS:** `.DS_Store` y `__MACOSX/` aparecen por todas partes; ignorarlos, no corregirlos como contenido.
- **`hw2/solution/hw2-solution.do` es un stub sin usar** (copia del de hw1, sobre funciones de producción) — ignóralo. La solución real de hw2 está en Python: `code/hw2_logit_nl.py` (Logit/Nested Logit) y `code/hw2_blp.py` (BLP, con `archive/demand_estimation/` como referencia original), volcada en `report/hw2-report.tex` y `rubric.yaml`. Ver `hw-report-explanation.md` para las diferencias documentadas entre el report reproducido y `hw2-solution.pdf` del profesor.
- Los datos de hw2 están en `raw/` como **`.csv`** (no `.dta` como dice el enunciado). `v.csv` son draws N(0,1): usar solo columnas 1-20.
