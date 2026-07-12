# hw2-report.pdf vs hw2-solution.pdf — qué usar para corregir

**Referencia de corrección: `report/build/hw2-report.pdf`** (reproducible con los datos distribuidos
`raw/*.csv` y validado por las entregas de los alumnos). `hw2-solution.pdf` (del profesor) sirve de
respaldo para la **lectura económica**, pero para los **números** de las matrices y de BLP manda el
report. Resumen de las diferencias:

## Dónde son idénticos (sin diferencia)
- Logit MCO (1a–1c) y Logit IV costo/Hausman (2a, 2c): coeficientes **exactos** en ambos.
- Elasticidades propias medias, 9 modelos (3a): idénticas.
- Nested Logit MCO (4b): idéntico.

## Diferencia 1 — Matrices de elasticidad farmacia 109 / semana 39 (3b, 4e)
- Los valores **celda a celda** del profesor **no se reproducen** con los `raw/*.csv` (probable
  re-indexación de mercados o datos de otra cosecha).
- `hw2-report.pdf` las calcula con la fórmula estándar sobre los datos entregados.
- **Validación:** Castillo, Álvarez y Sandoval —independientemente— obtuvieron *exactamente* la
  diagonal del report (p. ej. −1.907, −2.835, −4.080, …), no la del profesor (−2.66, −3.49, …).
- **Regla:** para 3b/4e se usa la matriz del **report**. Que una entrega "no calce con el PDF del
  profesor" **no** es motivo de descuento si calza con el report.

## Diferencia 2 — BLP: σ_G (coeficiente aleatorio del genérico)
| Parámetro | Profesor (solution.pdf) | Report | ¿Sustantivo? |
|---|---|---|---|
| α (precio, lineal) | −1.39 | −1.33 | No (~4%) |
| α_I (precio×ingreso) | 0.088 | 0.083 | No |
| **Efecto medio del precio** | **−0.46** | **−0.45** | No (casi igual) |
| Objetivo GMM | 363.1 | 362.7 | No (casi igual) |
| **σ_G (genérico)** | **0.78** | **0.39** | **Sí (~2×)** |

- σ_G está **débilmente identificado** (EE ≈ 1.4; el objetivo GMM es casi plano en σ_G). Ninguno es
  "incorrecto".
- La diferencia viene de **GMM 1 paso (σ_G≈0.39, report) vs 2 pasos (σ_G≈0.78, profesor)** y de los
  draws de simulación. Confirmado por alumnos: Castillo (1 paso → 0.39); Álvarez/Sandoval (2 pasos → 0.81).
- Lo económicamente relevante (efecto medio del precio, α_I>0, BLP rompe IIA con cruzadas más ricas) es
  **igual** en ambos. La matriz BLP difiere ~20–30% en magnitud como *consecuencia* de σ_G, con el
  mismo patrón.
- **Regla:** aceptar **ambas** versiones (1 paso ≈0.39 y 2 pasos ≈0.78–0.81). No penalizar por cuál se eligió.

## Nota sobre Nested Logit IV (4c/4d/4e) — dos especificaciones
- **Pauta/report (sigue el `.do` del profesor):** instrumenta **solo el precio** → precio ≈ +0.11,
  σ ≈ 1.10; MC2E ≈ MCO ⇒ conclusión: el instrumento de Hausman está casi colineal con el precio (precios
  uniformes a nivel nacional), aporta poca variación independiente.
- **Alternativa (más estándar):** instrumentar **precio y ln(s_jg)** (ambos endógenos) → precio ≈ −0.49,
  σ ≈ −2.96 ⇒ conclusión: el signo del precio se corrige pero σ∉[0,1) (identificación débil del nido).
- Las tres entregas de la muestra usaron la **alternativa**. Ambas llevan a "la identificación falla",
  pero por razones distintas. Decisión de criterio del TA: aceptar ambas o exigir la conclusión de la pauta.
