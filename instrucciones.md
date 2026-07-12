# Prompt para Claude Code: Generar informe de resultados de Tarea IO EmpÃ­rica

## Contexto
Soy TA de Empirical Industrial Organization. Necesito completar un informe de resultados (pauta de correcciÃ³n) a partir de:

## Estructura actual del proyecto
solution/
âââ src/
â âââ ChileAnalysis.dta # Base de datos original
âââ code/
â âââ hw1-solution.do # CÃ³digo Stata con todas las respuestas
âââ report/
â âââ hw1-report.tex # Enunciado YA COPIADO, listo para agregar respuestas
âââ bld/
â âââ fig/ # VacÃ­o (para figuras generadas)
â âââ tab/ # VacÃ­o (para tablas generadas)
âââ out/ # (opcional, para outputs finales)

## Tarea especÃ­fica

Quiero que **analices**:

1. **El cÃ³digo `code/hw1-solution.do`** - Identifica quÃ© resultados produce cada secciÃ³n del cÃ³digo
2. **El enunciado en `report/hw1-report.tex`** - Para saber quÃ© pregunta corresponde a cada resultado

## Output deseado

**NO necesito que reescribas el enunciado** (ya estÃ¡ en `hw1-report.tex`).

Solo necesito que **completes** `hw1-report.tex` agregando dentro de cada `\begin{solution} ... \end{solution}`:

### Contenido de cada solution:
- **Resultados clave**: Tablas y figuras relevantes cuando corresponda segÃºn enunciado o para complentar la respuesta
-  **Referencias a outputs**: Usar rutas relativas a `bld/`
  - Tablas: `\input{../bld/tab/nombre_tabla.tex}`
  - Figuras: `\includegraphics{../bld/fig/nombre_figura.pdf}`
- **InterpretaciÃ³n breve**: 2-3 lÃ­neas explicando el hallazgo principal

### Formato de ejemplo:
```latex
\begin{solution}
La regresiÃ³n de mÃ­nimos cuadrados ordinarios arroja los siguientes resultados:

\input{../bld/tab/regresion_mco.tex}

Se observa que el coeficiente de interÃ©s es positivo y estadÃ­sticamente significativo al 5\%, lo que sugiere una relaciÃ³n positiva entre X e Y.

Ver cÃ³digo: \texttt{code/hw1-solution.do} lÃ­neas 45-60.
\end{solution}
