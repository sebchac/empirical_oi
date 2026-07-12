# Pauta Tarea 2 — Estimación de demanda (Logit, Nested Logit, BLP)

Informe-pauta autocontenido: **`report/build/hw2-report.pdf`**.

## Estructura
- `code/hw2_logit_nl.py` — Logit (MCO e IV) y Nested Logit. → `bld/results_logit_nl.json`
- `code/hw2_blp.py` — BLP (coeficientes aleatorios, pyblp). → `bld/results_blp.json`
- `code/hw2_tables.py` — genera `bld/tab/*.tex` y `bld/fig/*.pdf`.
- `report/hw2-report.tex` — enunciado + 18 soluciones. Compila a `report/build/`.
- `hw2-solution.do` / `hw2-solution.pdf` — código y pauta originales del profesor (Stata).

Datos de entrada: `../raw/*.csv`. Mercado = `store`×`week`; tamaño = `count`; cuotas = `sales/count`.

## Reproducir
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt        # OJO: numpy<2 (pyblp 1.2.0)
python code/hw2_logit_nl.py
python code/hw2_blp.py
python code/hw2_tables.py
cd report && latexmk -pdf -output-directory=build hw2-report.tex
```

## Validación vs. pauta del profesor
- Logit (todos los coeficientes de precio y promoción) y Nested Logit (MCO e IV): **coinciden a 3–4 decimales**.
- Elasticidades propias medias (9 modelos): coinciden.
- BLP: α, α_I, efecto medio del precio y objetivo GMM (362.7 vs 363.1) coinciden estrechamente.

## Notas para la corrección (answer key)
- La **matriz de elasticidades 109/39** (preguntas 3b, 4e, BLP2) se reproduce sobre los CSV
  distribuidos; difiere por celda del PDF del profesor (esa parte no estaba en el `.do`; el PDF
  probablemente usó re-indexación de mercados o datos de otra cosecha). **Para calificar, use los
  valores de esta pauta**, que son los que obtendrán los estudiantes con `raw/*.csv`.
- BLP `σ_G` (genérico) ≈ 0.39 con los draws de `v.csv` (col. 1–20), vs 0.78 del PDF: el parámetro está
  débilmente identificado (objetivo plano / SE grande) y depende de los draws. Las conclusiones
  económicas no cambian.
- Nested Logit arroja precio positivo y σ̂>1 (instrumento débil): es el resultado correcto y esperado
  según la pauta; sus elasticidades no son interpretables.
