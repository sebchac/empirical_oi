"""
================================================================================
 PAUTA TAREA 2 - OI EMPIRICA  |  Generacion de tablas (bld/tab) y figuras (bld/fig)
================================================================================
 Lee bld/results_logit_nl.json y bld/results_blp.json y produce los fragmentos
 LaTeX (una tabla por subparte) y las figuras del informe.
 Convencion: notas envueltas en minipage (wrap); matrices anchas con \resizebox.
================================================================================
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLD = os.path.join(HERE, "..", "bld") + os.sep
TAB = BLD + "tab" + os.sep
FIG = BLD + "fig" + os.sep
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

R = json.load(open(BLD + "results_logit_nl.json"))
B = json.load(open(BLD + "results_blp.json"))

NAMES = ["Tylenol (25)", "Tylenol (50)", "Tylenol (100)", "Advil (25)", "Advil (50)",
         "Advil (100)", "Bayer (25)", "Bayer (50)", "Bayer (100)", "Generic (50)", "Generic (100)"]
SHORT = ["T25", "T50", "T100", "A25", "A50", "A100", "B25", "B50", "B100", "G50", "G100"]


def f3(x):
    return f"{x:.3f}"


def cell(x):  # evita -0.00
    return f"{0.0 if abs(x) < 0.005 else x:.2f}"


def table(fname, caption, colspec, body, note, resize=False, small=None):
    """body = filas entre \toprule y \bottomrule (incluye header, \midrule, datos)."""
    inner = f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{body}\\bottomrule\n\\end{{tabular}}"
    if resize:
        inner = "\\resizebox{\\linewidth}{!}{%\n" + inner + "}"
    size = f"\\{small}\n" if small else ""
    txt = f"""\\begin{{table}}[H]
\\centering
{size}\\caption{{{caption}}}
{inner}

\\smallskip
{{\\footnotesize\\begin{{minipage}}{{0.92\\linewidth}}{note}\\end{{minipage}}}}
\\end{{table}}"""
    with open(TAB + fname, "w") as f:
        f.write(txt.rstrip() + "\n")


# =============================================================== 1a/1b/1c  Logit MCO
def coef_rows(spec):
    out = ""
    for lab, key in [("Precio ($\\hat{\\alpha}$)", "price"), ("Promoci\\'on", "prom")]:
        c, se = R["logit"][spec][key], R["logit"][spec][key + "_se"]
        se_s = "---" if (not np.isfinite(se) or se == 0) else f"({f3(se)})"
        out += f"{lab} & {f3(c)} & {se_s} \\\\\n"
    return out


for spec, fname, cap, note in [
    ("ols1", "q1a_ols1.tex", "Logit MCO -- Modelo 1 (promociones)",
     "VD: $y=\\ln(s_j/s_0)$. EE robustos con \\texttt{cluster(store)}. $N=38{,}544$."),
    ("ols2", "q1b_ols2.tex", "Logit MCO -- Modelo 2 (promociones + dummies de marca)",
     "VD: $y=\\ln(s_j/s_0)$. Incluye 11 dummies de marca. EE \\texttt{cluster(store)}."),
    ("ols3", "q1c_ols3.tex", "Logit MCO -- Modelo 3 (promociones + dummies marca$\\times$farmacia)",
     "VD: $y=\\ln(s_j/s_0)$. Incluye dummies \\texttt{brand\\#store}. EE \\texttt{cluster(store)}."),
]:
    body = "& Coeficiente & EE \\\\\n\\midrule\n" + coef_rows(spec)
    table(fname, cap, "lcc", body, note)


# =============================================================== comparaciones de coeficientes
def compare_body(specs, labels):
    head = " & " + " & ".join(labels) + " \\\\\n\\midrule\n"
    pr = "Precio ($\\hat{\\alpha}$) & " + " & ".join(f3(R["logit"][s]["price"]) for s in specs) + " \\\\\n"
    pm = "Promoci\\'on & " + " & ".join(f3(R["logit"][s]["prom"]) for s in specs) + " \\\\\n"
    return head + pr + pm


table("q1d_ols_compare.tex", "Logit MCO: comparaci\\'on de los tres modelos", "lccc",
      compare_body(["ols1", "ols2", "ols3"], ["Modelo 1", "Modelo 2", "Modelo 3"]),
      "Modelo 1: prom. Modelo 2: + dummies de marca. Modelo 3: + dummies marca$\\times$farmacia.")
table("q2a_iv_cost.tex", "Logit IV (MC2E) -- instrumento: costo mayorista", "lccc",
      compare_body(["ivc1", "ivc2", "ivc3"], ["Modelo 1", "Modelo 2", "Modelo 3"]),
      "Precio instrumentado con \\texttt{cost}. EE \\texttt{cluster(store)}.")
table("q2c_iv_hausman.tex", "Logit IV (MC2E) -- instrumentos a la Hausman", "lccc",
      compare_body(["ivh1", "ivh2", "ivh3"], ["Modelo 1", "Modelo 2", "Modelo 3"]),
      "Precio instrumentado con el precio medio en otras farmacias (misma marca/semana). "
      "EE \\texttt{cluster(store)}.")

# Comparacion del precio en los 3 enfoques (apoyo a 2b/2d)
allivs = ("& Modelo 1 & Modelo 2 & Modelo 3 \\\\\n\\midrule\n"
          + "MCO & " + " & ".join(f3(R['logit'][s]['price']) for s in ['ols1', 'ols2', 'ols3']) + " \\\\\n"
          + "IV: costo & " + " & ".join(f3(R['logit'][s]['price']) for s in ['ivc1', 'ivc2', 'ivc3']) + " \\\\\n"
          + "IV: Hausman & " + " & ".join(f3(R['logit'][s]['price']) for s in ['ivh1', 'ivh2', 'ivh3']) + " \\\\\n")
table("q2_price_allivs.tex", "Coeficiente de precio seg\\'un estrategia de identificaci\\'on", "lccc",
      allivs, "Coeficiente de precio ($\\hat{\\alpha}$). Modelo 1: prom; 2: +marca; 3: +marca$\\times$farmacia.")

# =============================================================== 3a  Elasticidades medias 9 modelos
order = ["ols1", "ols2", "ols3", "ivc1", "ivc2", "ivc3", "ivh1", "ivh2", "ivh3"]
heads = ["MCO1", "MCO2", "MCO3", "IVc1", "IVc2", "IVc3", "IVh1", "IVh2", "IVh3"]
body = "Producto & " + " & ".join(heads) + " \\\\\n\\midrule\n"
for b in range(1, 12):
    body += f"{NAMES[b-1]} & " + " & ".join(f3(R["elas_mean"][m]["by_brand"][str(b)]) for m in order) + " \\\\\n"
body += "\\midrule\n\\textbf{Total} & " + " & ".join(f3(R["elas_mean"][m]["total"]) for m in order) + " \\\\\n"
table("q3a_elas_means.tex", "Elasticidad propia media por producto y modelo (9 modelos)", "l" + "c" * 9, body,
      "Elasticidad propia $\\eta_{jj}=\\hat{\\alpha}\\,p_j(1-s_j)$, promediada sobre farmacias y semanas. "
      "MCO1--3 y IVc/IVh1--3 corresponden a los Modelos 1--3 con costo (IVc) y Hausman (IVh).", small="small")


# =============================================================== matrices 11x11
def matrix_body(M):
    M = np.array(M)
    out = "& " + " & ".join(SHORT) + " \\\\\n\\midrule\n"
    for i in range(11):
        out += SHORT[i] + " & " + " & ".join(cell(M[i, j]) for j in range(11)) + " \\\\\n"
    return out


table("q3b_elas_matrix_logit.tex",
      "Matriz de elasticidades Logit (IV Hausman, Modelo 1) -- farmacia 109, semana 39",
      "l" + "r" * 11, matrix_body(R["elas_matrix"]["logit_ivh1"]),
      "Celda $(j,k)=\\dfrac{\\partial s_j}{\\partial p_k}\\dfrac{p_k}{s_j}$. Filas $=j$, columnas $=k$. "
      "Las cruzadas $\\approx 0$ por IIA y por el bien externo dominante.", resize=True)
table("q3b_elas_matrix_logit_brand.tex",
      "Matriz de elasticidades Logit (IV Hausman, Modelo 2 con dummies) -- farmacia 109, semana 39",
      "l" + "r" * 11, matrix_body(R["elas_matrix"]["logit_ivh2"]),
      "Con $\\hat{\\alpha}$ del modelo con dummies de marca; elasticidades propias mayores en magnitud.",
      resize=True)
table("q4e_nl_elas_matrix.tex",
      "Matriz de elasticidades Nested Logit (IV) -- farmacia 109, semana 39",
      "l" + "r" * 11, matrix_body(R["elas_matrix"]["nested_iv"]),
      "Estimaci\\'on NL-IV con $\\hat{\\alpha}>0$ y $\\hat{\\sigma}>1$: elasticidades con signo/escala "
      "an\\'omalos (ver comentario).", resize=True)
table("blp2_elas_matrix.tex",
      "Matriz de elasticidades BLP (coeficientes aleatorios) -- farmacia 109, semana 39",
      "l" + "r" * 11, matrix_body(B["elas_matrix_109_39"]),
      "Patr\\'on de sustituci\\'on m\\'as rico que el Logit: las cruzadas var\\'ian por columna seg\\'un "
      "cercan\\'ia en precio/marca.", resize=True)

# =============================================================== 4b/4c/4d  Nested Logit
nl = R["nested"]
nlb_ols = ("& Coeficiente & EE \\\\\n\\midrule\n"
           f"Precio ($\\hat{{\\alpha}}$) & {f3(nl['ols']['price'])} & ({f3(nl['ols_se']['price'])}) \\\\\n"
           f"$\\ln(s_{{j|g}})$ ($\\hat{{\\sigma}}$) & {f3(nl['ols']['l_sjg'])} & ({f3(nl['ols_se']['l_sjg'])}) \\\\\n"
           f"Promoci\\'on & {f3(nl['ols']['prom'])} & ({f3(nl['ols_se']['prom'])}) \\\\\n"
           f"Constante & {f3(nl['ols']['Intercept'])} & \\\\\n")
table("q4b_nl_ols.tex", "Nested Logit -- MCO (promociones, sin dummies)", "lcc", nlb_ols,
      "VD: $y=\\ln(s_j/s_0)$. EE \\texttt{cluster(store)}.")
nlb_iv = ("& Coeficiente & EE \\\\\n\\midrule\n"
          f"Precio ($\\hat{{\\alpha}}$) & {f3(nl['iv']['price'])} & ({f3(nl['iv_se']['price'])}) \\\\\n"
          f"$\\ln(s_{{j|g}})$ ($\\hat{{\\sigma}}$) & {f3(nl['iv']['l_sjg'])} & ({f3(nl['iv_se']['l_sjg'])}) \\\\\n"
          f"Promoci\\'on & {f3(nl['iv']['prom'])} & ({f3(nl['iv_se']['prom'])}) \\\\\n"
          f"Constante & {f3(nl['iv']['const'])} & \\\\\n")
table("q4c_nl_iv.tex", "Nested Logit -- MC2E (instrumentos Hausman y $n_{segm}$)", "lcc", nlb_iv,
      "Precio instrumentado con \\texttt{ivhausman} y $n_{segm}$. EE \\texttt{cluster(store)}.")
nlb_cmp = ("& MCO & MC2E \\\\\n\\midrule\n"
           f"Precio ($\\hat{{\\alpha}}$) & {f3(nl['ols']['price'])} & {f3(nl['iv']['price'])} \\\\\n"
           f"$\\ln(s_{{j|g}})$ ($\\hat{{\\sigma}}$) & {f3(nl['ols']['l_sjg'])} & {f3(nl['iv']['l_sjg'])} \\\\\n"
           f"Promoci\\'on & {f3(nl['ols']['prom'])} & {f3(nl['iv']['prom'])} \\\\\n")
table("q4d_nl_compare.tex", "Nested Logit: MCO vs MC2E", "lcc", nlb_cmp,
      "$\\hat{\\alpha}>0$ y $\\hat{\\sigma}>1$ en ambos casos (ver comentario sobre identificaci\\'on).")

# =============================================================== BLP1  Coeficientes Logit vs RC
beta = B["beta"]
lg = R["logit"]["ivh2"]
brand_betas = [beta[f"brand_{i}"] for i in range(1, 12)]
b1 = ("& Logit (IV) & Coef.\\ Aleatorios (BLP) \\\\\n\\midrule\n"
      f"Precio ($\\alpha$) & {f3(lg['price'])} & {f3(B['alpha_mean_price'])} \\\\\n"
      f"Promoci\\'on & {f3(lg['prom'])} & {f3(beta['prom'])} \\\\\n")
for i, nm in enumerate(NAMES):
    b1 += f"{nm} & --- & {f3(brand_betas[i])} \\\\\n"
b1 += ("\\midrule\n"
       f"$\\alpha$ (precio, medio) & {f3(lg['price'])} & {f3(B['alpha_mean_price'])} \\\\\n"
       f"$\\sigma_G$ (gen\\'erico, aleatorio) & --- & {f3(B['sigma_generic'])} \\\\\n"
       f"$\\alpha_I$ (precio$\\times$ingreso) & --- & {f3(B['pi_price_income'])} \\\\\n"
       f"Efecto medio del precio$^{{*}}$ & {f3(lg['price'])} & {f3(B['price_mean_effect'])} \\\\\n"
       f"Objetivo GMM & --- & {B['objective']:.1f} \\\\\n")
table("blp1_coef.tex", "Estimaci\\'on BLP: Logit (IV) vs.\\ Coeficientes Aleatorios", "lcc", b1,
      f"$^{{*}}$ Efecto medio del precio en BLP $=\\alpha+\\alpha_I\\bar{{I}}$, con $\\bar{{I}}={B['mean_income']:.2f}$. "
      "Logit (IV) = Modelo 2 con dummies de marca e instrumentos a la Hausman.", small="small")

# =============================================================== BLP3  Comparacion diagonales 109/39
diag_blp = np.diag(np.array(B["elas_matrix_109_39"]))
diag_lg = np.diag(np.array(R["elas_matrix"]["logit_ivh2"]))
diag_nl = np.diag(np.array(R["elas_matrix"]["nested_iv"]))
b3 = "Producto & Logit (IV) & Nested Logit & BLP \\\\\n\\midrule\n"
for i, nm in enumerate(NAMES):
    b3 += f"{nm} & {cell(diag_lg[i])} & {cell(diag_nl[i])} & {cell(diag_blp[i])} \\\\\n"
table("blp3_compare.tex",
      "Elasticidad propia por producto: Logit vs Nested Logit vs BLP (farmacia 109, semana 39)", "lccc", b3,
      "Logit: IV con dummies de marca. NL: estimaci\\'on IV (signos an\\'omalos por $\\hat{\\sigma}>1$).")

# =============================================================== FIGURAS
own_blp = [B["own_elas_by_brand"][str(b)] for b in range(1, 12)]
own_logit = [R["elas_mean"]["ivh2"]["by_brand"][str(b)] for b in range(1, 12)]
x = np.arange(11)
plt.figure(figsize=(8, 4))
plt.bar(x - 0.2, own_logit, 0.4, label="Logit (IV, dummies de marca)")
plt.bar(x + 0.2, own_blp, 0.4, label="BLP (coef. aleatorios)")
plt.xticks(x, SHORT, rotation=45)
plt.ylabel("Elasticidad propia media")
plt.title("Elasticidad propia media por producto")
plt.legend()
plt.tight_layout()
plt.savefig(FIG + "elas_comparacion.pdf")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, M, ttl in [(axes[0], np.array(R["elas_matrix"]["logit_ivh2"]), "Logit (IV)"),
                   (axes[1], np.array(B["elas_matrix_109_39"]), "BLP")]:
    im = ax.imshow(M, cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(11)); ax.set_xticklabels(SHORT, rotation=90, fontsize=7)
    ax.set_yticks(range(11)); ax.set_yticklabels(SHORT, fontsize=7)
    ax.set_title(f"Elasticidades {ttl} -- 109/39")
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(FIG + "heatmap_elasticidades.pdf")
plt.close()

print("Tablas:", len(os.listdir(TAB)), "| Figuras:", len(os.listdir(FIG)))
print("OK")
