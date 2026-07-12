---
name: pauta-tarea
description: Prepara la pauta-informe autocontenida de una tarea (homework) del curso de OI Empírica a partir del código y/o PDF de solución del profesor. Úsala cuando se pida "preparar/armar la pauta de hwN", generar el informe de resultados de una tarea, o reproducir los resultados correctos de una tarea. Trabaja sobre 2026_fall/01-homeworks/pbordon/hwN/solution/.
---

# Skill: preparar la pauta de una tarea

Produce un **informe-pauta autocontenido** para una tarea, siguiendo las convenciones de `CLAUDE.md`
(sección "Anatomía de una pauta", referencia `hw1/solution/`). El producto sirve como respuesta
modelo y como insumo de la skill `corregir-tarea`.

## Antes de empezar

Lee `CLAUDE.md` (estructura del repo, anatomía de la pauta, gotchas). Confirma con el TA **qué tarea**
(`hwN`) y **en qué formato** quiere el entregable final:
- **(A)** Solo extraer/consolidar los resultados correctos (si el profesor ya entregó una pauta PDF), o
- **(B)** Construir el informe LaTeX completo en el estilo del repo (`report/hwN-report.tex` + PDF).

Si hay ambigüedad real (alcance, software, qué preguntas reproducir), **pregunta antes de proceder**.

## Insumos (en `hwN/`)

| Insumo | Ubicación típica | Rol |
|---|---|---|
| Enunciado | `hwN/hw_N.pdf` | Define qué pide cada pregunta/subparte. Fuente de la numeración. |
| Código de solución | `solution/hwN-solution.do` (o `.py`) | Resultados correctos. Coef. esperados suelen ir en comentarios en línea. |
| Pauta del profesor | `solution/hwN-solution.pdf` (si existe) | **Fuente de la verdad** de resultados e interpretación económica. |
| Datos | `hwN/raw/*.csv` o `solution/src/*.dta` | Para reproducir. |

## Procedimiento

1. **Mapear preguntas → outputs.** Lee el enunciado y lista cada subparte (p. ej. 1a, 1b, …). Para
   cada una anota: qué se estima, método, variable dependiente, proxy/instrumento, y qué objeto se
   reporta (coeficiente, elasticidad, matriz, figura, comentario).
2. **Obtener los resultados correctos.**
   - Si existe `hwN-solution.pdf`: extrae de ahí los números canónicos y los SE reportados (son la
     referencia para tolerancias en la corrección). Transcríbelos con cuidado.
   - Si solo hay código: lee los comentarios en línea con los coeficientes; si faltan, **reprodúcelos**.
3. **Reproducir cuando haga falta** (este equipo **no tiene Stata**; sí Python y LaTeX):
   - Logit / Nested Logit (MCO y MC2E): `pandas` + `statsmodels`/`linearmodels` (IV2SLS, errores
     `cluster(store)`).
   - BLP (coeficientes aleatorios): `pyblp` (instalar si falta). Referencia lista para adaptar:
     `archive/demand_estimation/blp_nevo_ejemplo.py` y `blp.py`.
   - Datos hw2 en `raw/` como **CSV** (no `.dta`); `v.csv` son draws N(0,1), usar **columnas 1–20**.
4. **Generar outputs** en `solution/bld/tab/*.tex` y `solution/bld/fig/*.pdf`:
   - Tablas: fragmentos LaTeX autocontenidos con `booktabs` (`\toprule/\midrule/\bottomrule`),
     `\caption` y nota al pie (proxy/variable dependiente). Coeficientes a 3 decimales.
   - **Nombra cada archivo por subparte** para trazabilidad 1-a-1: `q1a_ols.tex`, `q2c_iv_hausman.tex`,
     `q4e_nl_elasticidades.tex`, `blp1_coeficientes.tex`, etc.
5. **Redactar el informe** `report/hwN-report.tex`:
   - Clase `\documentclass[answers]{exam}`, español (`babel`, `inputenc utf8`), `booktabs`.
   - El enunciado se copia tal cual; solo se rellena cada `\begin{solution} ... \end{solution}` con,
     en este orden: **(1) montaje** (1–2 líneas), **(2) resultado** (`\input{../bld/tab/...}` /
     `\includegraphics{../bld/fig/...}`, rutas relativas a `report/`), **(3) interpretación económica**
     (2–4 líneas con lectura económica real: signos, magnitudes, comparaciones, sesgos), **(4)
     trazabilidad** (`Ver código: \texttt{code/hwN-solution.do} líneas N--M.`).
   - Figuras que puedan no existir: patrón `\IfFileExists{...}{...}{\fbox{placeholder}}` (ver
     `hw1-report.tex` líneas 197–219) para que compile igual.
6. **Compilar:** `latexmk -pdf -outdir=report/build report/hwN-report.tex` (LaTeX disponible aquí).
7. **Verificar autocontención:** el PDF debe entenderse sin abrir nada más. Revisa que cada subparte
   del enunciado tenga su respuesta con respaldo.
8. **Generar `solution/rubric.yaml`** — la contraparte estructurada del informe, que consume la skill
   `corregir-tarea`. Por cada ítem (mismo id que el `\question`/`\part` del informe):
   - `resultado`: el/los valor(es) esperado(s) transcritos del paso 2/3, con `se` si la pauta lo
     reporta, y el `tipo` de tolerancia (`se_anchored` si hay SE; `pct_relativo` con un `pct`
     propuesto para cantidades derivadas sin SE — más ancho, `~15-20`, para BLP; `cualitativo` para
     matrices, derivaciones y comentarios sin un único valor puntual).
   - `puntos_teoricos`: **destila 2-4 bullets** del párrafo de interpretación económica que acabas de
     escribir en el `\begin{solution}` — son las ideas mínimas que la explicación de un alumno debe
     cubrir, no una paráfrasis completa de la respuesta modelo. El patrón de 4 partes que ya sigue el
     informe (montaje → resultado → interpretación → trazabilidad) hace esto casi mecánico: cada bullet
     sale de una oración de la interpretación.
   - `peso`: si la tarea tiene ponderaciones por sección, reparte el peso de la sección en partes
     iguales entre sus ítems; si no, usa pesos iguales para todos. Los pesos de toda la tarea deben
     sumar 1.
   - `notas_generales`: cualquier excepción conocida (especificaciones alternativas igualmente
     válidas, parámetros débilmente identificados, discrepancias entre lo reproducido y el PDF del
     profesor) — así `corregir-tarea` no las trata como errores del alumno.
   - Usa `2026_fall/01-homeworks/pbordon/hw2/solution/rubric.yaml` como plantilla concreta de formato.

## Salida

- `solution/bld/tab/` y `solution/bld/fig/` poblados.
- `solution/report/hwN-report.tex` con todas las soluciones.
- `solution/report/build/hwN-report.pdf` compilado.
- `solution/rubric.yaml` con todos los ítems, resultados/tolerancias y puntos_teoricos.
- Un resumen al TA: qué se reprodujo vs. qué se tomó del PDF del profesor, y cualquier discrepancia.

## Gotchas

- Nunca uses las rutas absolutas de Windows de los `.do` (`use "C:\Users\Paola Bordon\..."`); apunta a
  `raw/` o `src/` local.
- Todo en **español**, UTF-8.
- BLP es sensible a optimizador/semilla/integración: documenta semilla y método; espera diferencias
  pequeñas vs. la pauta del profesor (esto importa para fijar tolerancias en la corrección).
- Ignora `.DS_Store` y `__MACOSX/`.
