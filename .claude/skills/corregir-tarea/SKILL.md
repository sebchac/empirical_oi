---
name: corregir-tarea
description: Corrige las entregas de estudiantes de una tarea (homework) del curso de OI Empírica contra `rubric.yaml` de la pauta, y produce feedback por alumno + una planilla de notas. Úsala cuando se pida "corregir las entregas de hwN", "revisar las tareas", "poner notas a la tarea N" o generar la planilla de notas. Trabaja sobre `2026_fall/01-homeworks/<profesor>/hwN/submissions/` (cualquier profesor/tarea del curso — pbordon, jpazm, etc.), usando `2026_fall/01-homeworks/<profesor>/hwN/solution(es)/rubric.yaml` como fuente de verdad. Soporta entregas individuales o en pareja/grupo, y planilla de salida estándar o consolidada en un libro existente (ver campo `salida` de `rubric.yaml`).
---

# Skill: corregir las entregas de una tarea

Corrige cada entrega **contra `rubric.yaml`** de la tarea, evaluando dos ejes independientes —(a)
**resultado numérico** dentro de tolerancia y (b) **cobertura de los puntos teóricos** que la pauta
exige en la explicación— y genera feedback por alumno y la planilla de notas.

Esta skill es **genérica**: no contiene valores de ninguna tarea ni de ningún profesor en particular.
Todo lo específico de la tarea (ítems, valores esperados, SE, tolerancias, puntos teóricos, pesos, si
admite parejas, dónde se guarda la nota) vive en `rubric.yaml` de esa tarea — generado por la skill
`pauta-tarea`, o distilado directamente de una pauta PDF ya entregada por el profesor si `pauta-tarea`
no llega a construir un informe LaTeX nuevo. Si `rubric.yaml` no existe, créalo primero — no
improvises una rúbrica dentro de esta skill. La ubicación exacta de `rubric.yaml` varía según cómo
esté organizada la tarea (p. ej. `hwN/solution/` en pbordon, `hwN/solutions/` en jpazm) — el nombre
de carpeta no es fijo, léelo del layout real de la tarea.

## Filosofía de calificación (acordada con el TA)

**El resultado correcto manda; la explicación ajusta.** Cada ítem se puntúa 0–5:

1. **Base por exactitud del resultado**, calculada de forma **mecánica** (ver "Scorer determinístico"
   abajo) según el tipo de tolerancia que declara el ítem en `rubric.yaml`:
   - `se_anchored`: ancla la tolerancia al SE reportado en la pauta.
   - `pct_relativo`: cantidades derivadas sin SE (elasticidades medias, coeficientes multi-modelo);
     tolerancia en % relativo.
   - `cualitativo`: matrices, derivaciones y comentarios sin un único valor esperado — la base viene
     principalmente de la cobertura de puntos teóricos (paso 2).
   - **Signo correcto es obligatorio** en ítems con `se_anchored`/`pct_relativo`: signo equivocado
     limita la base a 0–1 aunque la magnitud "se acerque".
   - Sin intento / resultado ausente ⇒ 0–1.
   - BLP es sensible a optimizador/semilla/integración ⇒ tolerancia más amplia (`pct: 20` típico); se
     premia reproducir los patrones cualitativos (efecto medio del precio similar al Logit, mayor
     realismo en elasticidades cruzadas).
2. **Ajuste por la explicación**: si la cobertura de puntos teóricos (reportada por
   `verificador-explicacion`) es completa y sin errores, se **mantiene** la base. Si **omite un punto
   central**, se **baja** puntaje. Una cobertura excelente puede compensar parcialmente un resultado
   con desviación moderada (criterio del TA, no automático).

Las tolerancias concretas de cada tarea están **propuestas** en su `rubric.yaml`; revísalas con el TA
antes de una corrida grande y **no finalices notas silenciosamente**.

## Arquitectura

```
rubric.yaml (hwN/solution/)         ← fuente de verdad: ítems, valores, tolerancias, puntos_teoricos, pesos
        │
        ▼
corregir-tarea (esta skill, hilo principal — orquesta y es donde el TA puede intervenir)
  1. Lee rubric.yaml; si no existe, detente y pide correr `pauta-tarea` primero.
  2. Arma el roster (Cod Alumno ↔ Nombre) desde 2026_fall/grades.xlsx.
  3. Descomprime submissions/, ignora __MACOSX/ y .DS_Store.
  4. Por cada entrega (en lotes paralelos, p. ej. 4-6 a la vez):
       a. Despacha el agente `extractor-resultados`   → valor_extraído + texto_explicación por ítem
       b. Aplica el scorer determinístico (script)     → nota base 0-5 por tolerancia
       c. Despacha el agente `verificador-explicacion` → cobertura de puntos_teoricos, SIN el valor numérico
  5. Combina (b)+(c) por ítem, con la filosofía de arriba.
  6. Calibración: releé tú mismo (sin agentes) 2-3 entregas completas cruzando la distribución de
     notas, antes de cerrar nada — si encuentras una discrepancia real, releé todas.
  7. Escribe submissions/<id>/feedback_<id>.md + hwN_grades.xlsx + resumen de casos límite al TA.
```

Los dos agentes son de **solo lectura** y **no se comunican entre sí** — el verificador de explicación
nunca ve el valor numérico ni la nota base (evita que un buen resultado numérico "contamine" la lectura
de la explicación, y viceversa). Ver `.claude/agents/extractor-resultados.md` y
`.claude/agents/verificador-explicacion.md`.

## Procedimiento

1. **Cargar `rubric.yaml`.** Si `hwN/solution/rubric.yaml` no existe, avisa al TA y ofrece correr
   `pauta-tarea` primero — no inventes valores esperados ni tolerancias.
   - **Busca comunicaciones del profesor sobre severidad de corrección** en la carpeta
     `solution(es)/` (correos `.eml`, notas, README) antes de asumir que el peso teórico de
     `rubric.yaml` es la última palabra. Si el profesor comunicó explícitamente un criterio distinto
     (p. ej. "corrige muy suave", "todos deberían tener nota X") es **autoritativo por sobre tu
     propia lectura de la pauta** — documéntalo en `rubric.yaml` (`notas_generales`) con la fuente
     (quién, cuándo), la cita textual, y el **alcance explícito** (qué cubre y qué no: p. ej. "muy
     suave" no equivale automáticamente a perdonar una conclusión económica invertida — confírmalo
     con el TA en vez de asumirlo). Es específico de esa tarea/profesor: nunca generalices la
     instrucción a esta skill ni a otras tareas sin una instrucción equivalente del profesor
     correspondiente.
   - Si el profesor afirma haber dado "más código/material del habitual" o un scaffolding distinto
     al que ya tienes en `assignment/`, **verifica con diff/hash (`md5`) archivo por archivo** antes
     de aceptar la afirmación y cambiar tu juicio sobre qué es trabajo del alumno — puede tratarse
     del mismo material reenviado, no de scaffolding nuevo (pasó en jpazm/hw2: los archivos que
     parecían nuevos coincidían byte a byte con `assignment/`, y el PDF de soluciones tenía el mismo
     MD5 que el ya usado para construir la pauta).
2. **Roster:** mapea cada carpeta/zip de `submissions/` a `Cod Alumno` y `Nombre` usando
   `2026_fall/grades.xlsx` (hoja con R.U.N / Cod Alumno / Nombre). Los nombres de archivo de entrega
   suelen traer apellido y a veces el código; para casos ambiguos, pregunta al TA en vez de adivinar.
   - **Parejas/grupos** (`rubric.yaml` con `parejas_permitidas: true`): si el nombre de la
     carpeta/archivo trae dos nombres (p. ej. `alexandra-poncio-roxana-riquelme`), esa es **una**
     entrega que mapea a **dos** códigos de alumno — usa nombre completo (no solo apellido) para
     desambiguar entre alumnos con apellidos comunes. Propón la tabla completa
     carpeta → Cod Alumno(s) al TA **antes** de seguir; no la des por buena sin confirmar. Cada
     integrante recibe exactamente los mismos puntajes en la planilla final, y el `feedback.md` de
     esa entrega debe identificar a **todos** los integrantes (Cod Alumno + Nombre completos) en el
     encabezado, no solo a uno — cualquiera de los dos puede escribir preguntando por su nota y debe
     reconocerse en el documento.
3. **Por entrega:**
   - Descomprime el `.zip` si aplica. Ignora `__MACOSX/` y `.DS_Store` — no son contenido del alumno.
   - Despacha `extractor-resultados` (herramienta `Agent`) pasándole la carpeta de la entrega + la
     lista de ítems de `rubric.yaml` (solo `id`, `pregunta`, `resultado.tipo` — nunca los valores
     esperados). Recibe de vuelta `valor_extraído` + `texto_explicación` por ítem.
   - Corre el **scorer determinístico** (snippet abajo) sobre cada `valor_extraído` contra el bloque
     `resultado` del ítem correspondiente en `rubric.yaml` → nota base 0-5 por ítem.
   - Despacha `verificador-explicacion` pasándole, por ítem, **solo** `puntos_teoricos` (de
     `rubric.yaml`) + `texto_explicación` (del extractor) — nunca la nota base ni el valor esperado.
     Recibe de vuelta cobertura presente/ausente/incorrecto por punto, con evidencia.
     **Pasa el `texto_explicación` completo que devolvió el extractor, no un resumen o cita curada
     por ti** — condensar la explicación antes de pasarla es exactamente lo que causó una subcaptura
     sistemática de cobertura en 9 de 12 entregas de jpazm/hw1 (un supuesto "patrón de clase" que
     resultó ser, en la mayoría de los casos, contenido real que el resumen intermedio no transmitió).
     Si el texto del extractor es largo, es preferible eso a perder contenido real.
   - Puedes despachar varias entregas en paralelo (una llamada `Agent` por entrega por rol) para no
     serializar toda la corrección; un lote de 4-6 entregas a la vez es razonable para no saturar.
4. **Combinar y puntuar** cada ítem 0–5: parte de la nota base del scorer, ajusta según cobertura de
   explicación con la filosofía de arriba, y anota una **justificación breve** citando la evidencia que
   entregó `verificador-explicacion`.
   - **Criterio de flexibilidad (sin tocar la fórmula del punto 7):** no infles la nota subiendo un
     puntaje solo porque falta poco — eso es regalar puntaje. Sube la clasificación de un punto
     teórico puntual (de `ausente`/parcial a `presente`) únicamente con evidencia concreta:
     (a) el punto está verificablemente resuelto en código/datos propios del alumno (no en el script
     `assignment/` sin modificar) aunque no esté narrado en el informe — ver la excepción homónima en
     `verificador-explicacion.md`; o (b) es el **único** punto que falta en el ítem y, comparando entre
     entregas de la misma tarea, casi toda la clase omitió exactamente ese mismo punto — señal de que
     el enunciado dejaba espacio a interpretación, no de una debilidad individual. Ninguna de las dos
     excepciones aplica a un punto marcado `incorrecto` (mecanismo o conclusión que invierte la
     relación/causalidad esperada): ese puntaje se mantiene bajo sin importar cuánto se haya
     desarrollado alrededor — es exactamente el tipo de error que el TA quiere seguir penalizando con
     dureza.
   - **"Patrón de clase" (b): consulta al TA antes de decidir, y verifica contra texto completo, no
     contra resúmenes.** Nunca decidas unilateralmente si un patrón transversal se excusa o se
     penaliza — antes de puntuar, preséntale al TA la lista concreta de qué explicaciones se están
     omitiendo, ítem por ítem y con evidencia, y que el TA juzgue caso a caso si la omisión es
     *sustancial* (se penaliza igual para todos, incluso siendo transversal) o *ambigüedad menor del
     enunciado* (se puede excusar con un bono) — "todos lo omiten, no es culpa individual" aplicado
     en automático, sin distinguir severidad, hace que la pauta pierda su valor. Incluso si el TA
     aprueba excusar un patrón, un mismo alumno no debe beneficiarse de más de **un bono de patrón de
     clase por tarea** — 3-4 bonos simultáneos en el mismo alumno ya no describen una ambigüedad
     puntual del enunciado, describen que ese alumno no profundiza en el mecanismo de forma
     consistente. **Antes de declarar que un patrón es real, confirma que la evidencia de omisión
     viene de leer el texto completo de cada alumno, no de un resumen/cita ya curado por quien
     corrige** — en jpazm/hw1 un supuesto "patrón de clase" en dos ítems resultó ser, en 9 de 12
     entregas, un problema de que el resumen pasado a `verificador-explicacion` no capturaba pasajes
     que sí estaban completos en el informe (ver `notas_generales` de
     `jpazm/hw1/solutions/rubric.yaml`). Si dudas de si un patrón es real o un artefacto de resumen,
     relee tú mismo el texto completo de 2-3 entregas antes de presentárselo al TA.
   - **Ajustes generales (una "curva", no una reclasificación puntual con evidencia):** solo se
     aplican cuando hay una instrucción explícita y documentada que los respalde (del TA o del
     profesor — ver "Busca comunicaciones del profesor" en el paso 1), nunca por iniciativa propia
     de un agente. Incluso con esa instrucción, el ajuste **nunca** sube: (a) ítems "muy
     incompletos" (sección en blanco o casi, sin ningún desarrollo — un piso general no es lo mismo
     que compensar ausencia total), ni (b) ítems marcados `incorrecto`. Antes de escribir cualquier
     archivo, **simula el ajuste sobre toda la tarea y muéstrale al TA la tabla completa
     nota-anterior-vs-nueva** (no solo describas la regla en palabras) — calibrar una curva a
     ciegas lleva a sobre- o sub-corregir; enseñar los números concretos es lo que le permite al TA
     afinar el criterio con precisión.
5. **Calibración obligatoria antes de cerrar notas — releer directamente una muestra, sin agentes.**
   Antes de escribir cualquier `feedback.md` o planilla, elige **2-3 entregas** (con `Read`, tú mismo,
   el documento completo — no un sub-agente ni el resumen del extractor) y confirma que la cobertura
   de puntos teóricos que reportó `verificador-explicacion` coincide con lo que el informe realmente
   dice. **La muestra debe cruzar la distribución de notas** (al menos una baja, una media, una alta)
   — no alcanza con revisar solo las notas bajas: en jpazm/hw1 la subcaptura apareció también en una
   nota media-alta (6.0/7.0), no solo en las más bajas. Si la muestra no encuentra discrepancias,
   sigue con confianza. **Si encuentra aunque sea una discrepancia real** (contenido que el pipeline
   marcó ausente pero que el documento completo sí contiene), no la trates como un caso aislado:
   releé directamente **las entregas restantes** antes de cerrar cualquier nota — eso fue exactamente
   lo que ocurrió en jpazm/hw1 (3 de las primeras 3 entregas releídas tenían el problema, lo que llevó
   a releer las 12 completas y terminó corrigiendo 9 al alza). El costo de este paso es releer unos
   pocos informes de más; el costo de saltárselo es cerrar notas oficiales sobre datos no confiables.
   Documenta en el resumen al TA (paso 9) qué entregas se releyeron directamente y si hubo o no
   discrepancias.
6. **Feedback por alumno:** escribe `submissions/<id>/feedback_<id>.md` con, por ítem: valor extraído
   vs. esperado (dentro/fuera de tolerancia), cobertura de puntos teóricos (qué faltó, con cita), y
   puntaje con justificación.
7. **Notas:** `Total_% = Σ (score_item/5 · peso_item) · 100` (los `peso_item` de `rubric.yaml` suman 1
   por tarea); `Nota = round(1 + Total_%/100 · 6, 1)` (escala chilena 1.0–7.0).
8. **Planilla.** Por defecto, un archivo nuevo `2026_fall/hwN_grades.xlsx`, hoja `hoja_notas` (de
   `rubric.yaml`), columnas `Cod Alumno, Nombre, <ids de items en el orden de rubric.yaml>, Total %,
   Nota` — genera las columnas de ítems leyendo `rubric.yaml`, nunca las asumas fijas (una tarea con
   subpartes tipo hw2 y una con ítems planos tipo hw1 usan el mismo generador).
   - **Salida configurable:** si `rubric.yaml` trae un bloque `salida` (libro/hoja/columna_id
     distintos del default — p. ej. para consolidar en un libro de notas ya existente en vez de
     crear uno nuevo), **antes de escribir nada** abre ese libro y lee una hoja hermana ya
     existente (la que indique `estilo_referencia`) para replicar su estructura de columnas exacta
     — no asumas el formato estándar de arriba. Si el libro/hoja de destino no existe todavía,
     créalos con esa misma estructura inferida. Nunca sobrescribas una hoja que ya tenga datos sin
     confirmar con el TA.
9. **Entrega al TA:** la planilla + un resumen de casos límite (entregas incompletas, sin código,
   resultados muy desviados, ítems donde `verificador-explicacion` marcó "incorrecto") + qué entregas
   se releyeron directamente en el paso 5 de calibración y si hubo discrepancias, para que el TA
   confirme antes de cerrar notas.

## Scorer determinístico (aritmética, no criterio de un LLM)

La tolerancia se calcula con código, no con juicio de un agente — evita errores de cálculo del tipo
"¿0.68 cae dentro de ±1 SE de 0.693?". Snippet base a adaptar según el `tipo`/`tolerancia` del ítem:

```python
def score_tolerancia(valor_extraido, resultado):
    """resultado = bloque `resultado` de un ítem de rubric.yaml. Devuelve nota base 0-5, o None
    si el ítem es cualitativo (matriz/comentario/derivación) y la base debe venir de la cobertura
    de explicación en vez de un valor puntual."""
    if valor_extraido in (None, "no encontrado"):
        return 0  # o 1 si hubo un intento parcial visible en el texto

    tol = resultado.get("tolerancia")
    if tol == "cualitativo":
        return None  # la skill usa la cobertura de verificador-explicacion como base

    # una sola variable con SE (o la primera de `variables` si el ítem trae varias)
    esperado = resultado.get("valor_esperado")
    se = resultado.get("se")
    if esperado is None and "variables" in resultado:
        # el llamador debe invocar esta función una vez por variable y promediar/tomar la más baja
        raise ValueError("pasar una variable a la vez para items con 'variables'")

    if (esperado < 0) != (valor_extraido < 0) and esperado != 0:
        return 1  # signo incorrecto: penalización fuerte, tope 1

    if tol == "se_anchored" and se:
        z = abs(valor_extraido - esperado) / se
        thresholds = [(1, 5), (2, 4), (3, 3), (4, 2)]
    elif tol == "pct_relativo":
        pct = resultado.get("pct", 10) / 100
        z = abs(valor_extraido - esperado) / abs(esperado) / pct  # z=1 al borde de la tolerancia
        thresholds = [(1, 5), (2, 4), (3, 3), (4, 2)]
    else:
        return None

    for bound, score in thresholds:
        if z <= bound:
            return score
    return 1
```

Para ítems con varias `variables` (p. ej. precio + promoción en la misma tabla), corre la función por
variable y usa el **mínimo** como nota base del ítem (el peor componente manda, consistente con "el
resultado correcto manda").

Para ítems `tipo: comentario` o `tipo: derivacion` (sin `resultado.valor_esperado`), la nota base ES
la cobertura de `verificador-explicacion` traducida a 0-5 (p. ej. proporción de puntos `presente` ×5,
con signo/conclusión cualitativa correcta como requisito para no capar en 1-2).

## Snippet para escribir la planilla (openpyxl)

**Importante:** `Total %` y `Nota` se escriben como **fórmulas de Excel** (no como el número ya
calculado en Python), para que el profesor pueda auditar el cálculo abriendo la celda. El valor en
Python solo sirve para detectar casos límite antes de cerrar notas (paso "no finalices notas
silenciosamente"); nunca se vuelca como constante en `Total %`/`Nota`.

```python
import openpyxl, yaml
from openpyxl.utils import get_column_letter

rubric = yaml.safe_load(open(ruta_rubric))  # p. ej. "hwN/solution/rubric.yaml" o "hwN/solutions/rubric.yaml"
items = [it["id"] for it in rubric["items"]]
pesos = [it["peso"] for it in rubric["items"]]
salida = rubric.get("salida")  # None => modo estándar (archivo nuevo hwN_grades.xlsx)

def fila_scores(est):
    return [est["scores"].get(i, 0) for i in items]

def formulas(item_col_start, row):
    # item_col_start: columna (1-indexada) de la primera columna de ítem (C en modo estándar, B en consolidado)
    total_terms = [
        f"{get_column_letter(item_col_start + k)}{row}/5*{peso}"
        for k, peso in enumerate(pesos)
    ]
    total_col = get_column_letter(item_col_start + len(items))
    total_formula = f"=ROUND(({'+'.join(total_terms)})*100,1)"
    nota_formula = f"=ROUND(1+{total_col}{row}/100*6,1)"
    return total_formula, nota_formula

if salida is None:
    # Modo estándar: archivo nuevo, una fila por alumno con Cod Alumno + Nombre.
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = rubric["hoja_notas"]
    ws.append(["Cod Alumno", "Nombre", *items, "Total %", "Nota"])
    item_col_start = 3  # A=Cod Alumno, B=Nombre, C.. = items
    for r, est in enumerate(calificaciones, start=2):  # est = {"cod":..., "nombre":..., "scores":{item_id:0-5}}
        ws.append([est["cod"], est["nombre"], *fila_scores(est), None, None])
        total_f, nota_f = formulas(item_col_start, r)
        ws.cell(row=r, column=item_col_start + len(items)).value = total_f
        ws.cell(row=r, column=item_col_start + len(items) + 1).value = nota_f
    wb.save(f"2026_fall/{rubric['tarea']}_grades.xlsx")
else:
    # Modo consolidado: nueva hoja en un libro existente, replicando el estilo de una hoja hermana
    # (leída antes de escribir — ver paso 8). columna_id suele ser "ID" (sin Nombre aparte).
    wb = openpyxl.load_workbook(salida["libro"])
    ws = wb.create_sheet(salida["hoja"])
    ws.append([salida["columna_id"], *items, "Total %", "Nota"])
    item_col_start = 2  # A=ID, B.. = items
    for r, est in enumerate(calificaciones, start=2):  # una fila por CADA integrante si hubo parejas (mismos scores)
        ws.append([est["cod"], *fila_scores(est), None, None])
        total_f, nota_f = formulas(item_col_start, r)
        ws.cell(row=r, column=item_col_start + len(items)).value = total_f
        ws.cell(row=r, column=item_col_start + len(items) + 1).value = nota_f
    wb.save(salida["libro"])
```

## Gotchas

- `.gitignore` excluye `*.zip` y `*.xlsx`: la planilla y las entregas **no se versionan** (es
  deliberado).
- Entregas desordenadas: nombres heterogéneos, carpetas por ID o por nombre (individual o pareja),
  PDF suelto sin zip, a veces PDF **y** zip para la misma entrega (usa el zip como fuente primaria,
  el PDF como respaldo — no cuentes dos veces), carpetas espurias que no son entregas (revisa antes
  de asumir que todo lo que hay en `submissions/` es una entrega válida).
- Si la tarea admite parejas, confirma que el total de entregas × personas por entrega cuadra con el
  roster completo del curso (ni de más ni de menos) antes de escribir la planilla.
- Software mixto: Stata, Python, R — puede variar entre alumnos y entre preguntas de un mismo alumno;
  `extractor-resultados` está diseñado para eso, no asumas uno solo.
- **Resguardo "documento equivocado":** ya ocurrió un reclamo justificado de un alumno por puntaje 0
  en una pregunta que sí había respondido, porque se revisó el documento equivocado de su entrega
  (p. ej. un borrador en vez de la versión final, cuando había más de un PDF). Antes de puntuar 0 o
  "no encontrado" en cualquier ítem, confirma que `extractor-resultados` enumeró y revisó **todos**
  los documentos candidatos de esa entrega, no solo el primero que abrió.
- Diferencias numéricas atribuibles al software (Stata vs. Python vs. R) caen dentro de la tolerancia
  declarada; no las penalices aparte.
- Revisa siempre `notas_generales` en `rubric.yaml` — documentan excepciones conocidas (especificaciones
  alternativas válidas, parámetros débilmente identificados, discrepancias fuente-vs-pauta-profesor)
  que no deben tratarse como errores del alumno.
- Todo el feedback en **español**.

## Cross-references

- `.claude/skills/pauta-tarea/SKILL.md` — genera `rubric.yaml`; corre esta skill primero si falta.
- `.claude/agents/extractor-resultados.md` — recupera valores y texto de explicación, por entrega.
- `.claude/agents/verificador-explicacion.md` — juzga cobertura de puntos teóricos, en contexto fresco.
- `2026_fall/01-homeworks/pbordon/hw2/solution/rubric.yaml` — primer caso de uso real del esquema
  (ítems con SE, un solo autor por entrega, salida estándar).
- `2026_fall/01-homeworks/jpazm/hw2/solutions/rubric.yaml` — segundo caso de uso: `parejas_permitidas`
  y bloque `salida` apuntando a una hoja nueva dentro de un libro de notas consolidado ya existente.
  También es el ejemplo de referencia de `notas_generales` documentando una instrucción de severidad
  del profesor recibida por correo a mitad de la corrección (fuente, fecha, cita y alcance explícito).
