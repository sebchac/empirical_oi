# Tarea 2 — Criterio de notas y respaldo de la corrección

Documento de respaldo de las notas en `2026_fall/hw2_grades.xlsx`. Acompaña a los
`feedback_<cod>.md` por alumno (en `_grading/<cod>/`) y a
[`solution/hw-report-explanation.md`](solution/hw-report-explanation.md).

## 1. Referencia de corrección (answer key)

`solution/report/build/hw2-report.pdf`, **reproducible** con los datos distribuidos (`raw/*.csv`) y
validado por las propias entregas. Para las matrices 109/39 y el BLP manda este report (no el PDF del
profesor); detalle en `hw-report-explanation.md`.

## 2. Rúbrica

- Cada subparte (18 en total) se puntúa en enteros **0–5** (0 = no respondido / sin intento).
- **Nota ponderada por sección:** `Nota = 1 + 6 · Σ_sección [ peso · (promedio_ítems / 5) ]`.

| Sección | Peso | Ítems |
|---|---|---|
| Logit (MCO) | 15% | 1a, 1b, 1c, 1d |
| Logit IV | 15% | 2a, 2b, 2c, 2d |
| Elasticidades | 10% | 3a, 3b |
| Nested Logit | 25% | 4a, 4b, 4c, 4d, 4e |
| BLP | 35% | BLP1, BLP2, BLP3 |

## 3. Decisiones de calibración (acordadas con el TA)

1. **Result-first con tolerancia:** coeficientes dentro de ~±1 EE de la pauta → completo; elasticidades
   ±10%; BLP ±15–20% (débilmente identificado). Signo correcto obligatorio.
2. **Parte 1 (Logit MCO/IV, elasticidades):** todos los que entregaron reprodujeron los valores
   **exactos** de la pauta → 5 en esos ítems numéricos.
3. **Matrices 109/39:** se evalúan contra `hw2-report.pdf`. Que no calcen con el PDF del profesor **no**
   es motivo de descuento.
4. **Nested Logit (especificación):** se aceptan ambas: instrumentar solo el precio (como la pauta) o
   instrumentar precio y `ln(s_jg)` (más estándar; lo que usaron todos). 4c y 4e se evalúan por el
   resultado de la especificación elegida.
5. **4d (interpretación NL):** decisión del TA "exigir la lectura de la pauta" → se descuenta (5→3) a
   quien no señale que el instrumento de Hausman queda casi colineal con el precio (MC2E≈MCO).
   **Ver §6, punto abierto.**
6. **BLP:** se aceptan GMM de 1 paso (σ_G≈0.39) y de 2 pasos (σ_G≈0.78–0.81); se penalizan errores de
   magnitud o conclusiones equivocadas.

## 4. ¿Cómo se consideró la calidad de la explicación?

La nota combina **resultado** y **explicación**, pero su peso difiere por ítem:

- **Ítems numéricos** (1a–1c, 2a, 2c, 3b, 4b, 4c, 4e, BLP1, BLP2): dominados por el resultado correcto.
  Como los datos fuerzan los mismos números, casi todos obtuvieron 5.
- **Ítems de interpretación** (1d, 2b, 2d, 3a, 4d, BLP3): aquí pesa la explicación. Las deducciones por
  **confusiones** (según relevancia) aplicadas fueron:
  - **4d = 3 (universal):** ningún alumno señaló la colinealidad Hausman-precio de la pauta.
  - **Sandoval — BLP2/BLP3:** elasticidades BLP ~10× bajas (error de cómputo) → conclusión equivocada
    ("BLP más inelástico"). BLP2=2, BLP3=2.
  - **Poncio — BLP1/BLP3:** no reporta los parámetros BLP en el informe ("en el código"); BLP3 ordena
    mal los modelos (afirma que el Nested Logit es el de sustitución más rica). BLP1=3, BLP3=3.
  - **Flores R. y Muñoz Sofía — BLP ausente:** BLP1/2/3 = 0.

**Lectura honesta de las notas altas:** el grupo en 6.9 refleja trabajo genuinamente completo y
correcto (Parte 1 forzada por los datos + interpretaciones sólidas), no una corrección laxa. La
diferenciación real está en las colas (incompletos y con confusiones). No se "fabricaron" descuentos
donde la explicación era correcta.

## 5. Resumen por estudiante

| Cod | Estudiante | Nota | Síntesis |
|---|---|---|---|
| 2026830479 | Molina | 6.9 | Completo; BLP3 muy bueno |
| 2026830480 | Altamirano | 6.9 | Muy completo (robustez); tardía |
| 2025829049 | Álvarez | 6.9 | Completo; anexo de matrices |
| 2026830484 | Caro | 6.9 | Completo |
| 2026830485 | Castillo | 6.9 | Completo |
| 2026830486 | Flores Enrique | 6.9 | Completo (PDF+ZIP) |
| 2025827317 | Flores Roberto | 4.8 | LATE + sin BLP; menor: Modelo 2 / N |
| 2026830488 | Medina | 6.9 | Muy completo |
| 2026830489 | Moreno | 6.9 | Muy completo |
| 2026831676 | Muñoz Lombardi | 6.9 | Completo (R + Python) |
| 2026832010 | Muñoz Chaparro | 6.9 | Completo; pipeline ordenado; tardía |
| 2025825782 | Muñoz Sofía | 4.8 | Sin BLP |
| 2026830493 | Poncio | 6.3 | BLP1 no reportado; BLP3 confuso |
| 2026830494 | Reyes | 6.9 | Completo |
| 2026830495 | Riquelme | 6.9 | Completo |
| 2026830496 | Sandoval | 5.9 | Error de magnitud en BLP2/BLP3 |
| 2026830497 | Valenzuela | 6.9 | Completo; verificar mercado 109 |

## 6. Puntos abiertos (honestidad para defender ante la profesora)

- **El descuento de 4d es discutible.** La "colinealidad Hausman-precio" de la pauta es propia de la
  especificación *price-only*. Como **todos** instrumentaron precio **y** `ln(s_jg)`, en sus
  estimaciones la **primera etapa es fuerte** (F enormes) y el problema correcto a señalar es
  **estructural** (σ fuera de [0,1)). Varios lo diagnosticaron bien (Altamirano, Muñoz Chaparro,
  Molina). Por tanto, exigir la frase de la pauta podría **penalizar un razonamiento correcto**. Si se
  revisa este criterio (4d=5 a quien diagnostique bien el problema estructural), la mayoría subiría a
  7.0.
- **Profundidad de revisión:** lectura completa de 5 entregas (las 3 de calibración + las 2 tardías) y
  revisión dirigida (Parte 1, NL-4d, BLP) del resto sobre el texto de cada informe. No se hizo una
  auditoría línea por línea de cada comentario.
- **Casos a confirmar:** Flores R. (descuento por atraso, lo define el TA); Valenzuela (mercado 109 en
  sus matrices).
