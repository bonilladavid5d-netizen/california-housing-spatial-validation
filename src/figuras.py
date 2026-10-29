"""Cuatro figuras que responden la pregunta del caso. Color = modelo en todas las figuras."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from .modelos import NOMBRES

COLOR = {"mco": "#2a78d6", "rf": "#eb6834", "gb": "#1baf7a", "media": "#8a8984"}
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
DISENOS = {"holdout_original": "Split original 80/20",
           "aleatorio_5f": "A. Aleatorio, 5 folds",
           "espacial_5f": "B. Espacial, 5 folds + margen 10 km"}

ETIQUETA_BANDA = {"[0, 1)": "0–1", "[1, 2)": "1–2", "[2, 3)": "2–3", "[3, 4)": "3–4", "[4, 5)": "4–5",
                  "[5, 5.00001)": "= 5", "techo (5,00001)": "techo"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2,
    "ytick.color": TINTA_2, "axes.titlecolor": TINTA, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _miles(v):
    return f"{v / 1000:,.0f} mil".replace(",", ".")


def fig_rmse(met, cfg, destino):
    usd = cfg["datos"]["unidad_objetivo_usd"]
    orden = ["media", "mco", "gb", "rf"]
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    for ax, (d, titulo) in zip(ejes, DISENOS.items()):
        g = met[(met["diseno"] == d) & (met["nivel"] == "global")].set_index("modelo")
        f = met[(met["diseno"] == d) & (met["nivel"] == "fold")]
        for k, m in enumerate(orden):
            v = g.loc[m, "rmse"] * usd
            ax.barh(k, v, height=0.55, color=COLOR[m])
            if d != "holdout_original":
                r = f[f["modelo"] == m]["rmse"] * usd
                ax.plot([r.min(), r.max()], [k, k], color=TINTA, lw=1.2)
                ax.plot([r.min(), r.max()], [k, k], "|", color=TINTA, ms=8)
            x_texto = v if d == "holdout_original" else max(v, f[f["modelo"] == m]["rmse"].max() * usd)
            r2 = f"{g.loc[m, 'r2'] + 0.0:.3f}".replace(".", ",").replace("-", "−").replace("−0,000", "0,000")
            ax.text(x_texto + 2500, k, f"{_miles(v)}  (R² {r2})", va="center", fontsize=8.5, color=TINTA)
        ax.set_title(titulo, loc="left")
        ax.set_yticks(range(len(orden)), [NOMBRES[m] for m in orden])
        ax.set_xlim(0, 175000)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}")
        ax.grid(axis="x", color=REJILLA, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("RMSE global, miles de USD de 1990")
    fig.suptitle("Error de predicción (RMSE) por modelo y diseño de evaluación", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=TINTA, y=1.06)
    fig.text(0.01, 0.97, "Barra: RMSE sobre todas las predicciones fuera de muestra. Línea: rango de RMSE entre "
             "folds. R² entre paréntesis.", fontsize=9, color=TINTA_2)
    fig.savefig(destino / "fig1_rmse_por_diseno.png")
    plt.close(fig)


def fig_mapa(cel, cfg, destino):
    usd = cfg["datos"]["unidad_objetivo_usd"]
    tam = cfg["segmentos"]["celda_mapa_grados"]
    c = cel[(cel["diseno"] == "espacial_5f") & (cel["n"] >= cfg["segmentos"]["n_minimo_celda_mapa"])]
    vmax = np.ceil(np.percentile(c[["rmse_mco", "rmse_rf"]].to_numpy() * usd, 98) / 10000) * 10000
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap("Blues")
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, m in zip(ejes, ["mco", "rf"]):
        for _, r in c.iterrows():
            ax.add_patch(Rectangle((r["lon_centro"] - tam / 2, r["lat_centro"] - tam / 2), tam, tam,
                                   facecolor=cmap(norm(r[f"rmse_{m}"] * usd)), edgecolor="white", lw=0.8))
        ax.set_xlim(-124.6, -113.9)
        ax.set_ylim(32.3, 42.2)
        ax.set_aspect(1 / np.cos(np.radians(37)))
        rmse_mostradas = np.sqrt((c["n"] * c[f"rmse_{m}"] ** 2).sum() / c["n"].sum()) * usd
        ax.set_title(NOMBRES[m], loc="left", pad=22)
        ax.text(0, 1.015, f"RMSE de las celdas mostradas: {_miles(rmse_mostradas)} USD", transform=ax.transAxes,
                fontsize=9, color=TINTA_2, va="bottom")
        ax.set_xlabel("Longitud")
        ax.grid(color=REJILLA, lw=0.6)
        ax.set_axisbelow(True)
    ejes[0].set_ylabel("Latitud")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ejes, shrink=0.8, pad=0.02)
    cb.set_label("RMSE de la celda, miles de USD de 1990")
    cb.formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}")
    cb.update_ticks()
    cb.outline.set_visible(False)
    fig.suptitle("Dónde se concentra el error con validación espacial (celdas de 0,5°)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=TINTA, y=0.98)
    fig.text(0.01, 0.92, f"Celdas con al menos {cfg['segmentos']['n_minimo_celda_mapa']} block groups. "
             "Cada celda se predice sin entrenar con su propia área ni con puntos a menos de 10 km.",
             fontsize=9, color=TINTA_2)
    fig.savefig(destino / "fig2_mapa_error_espacial.png")
    plt.close(fig)


def fig_bandas(seg, cfg, destino):
    s = seg[(seg["diseno"] == "espacial_5f") & (seg["tipo_segmento"] == "banda_valor")]
    bandas = list(dict.fromkeys(s.sort_values("media_observado")["segmento"]))
    modelos = ["mco", "gb", "rf"]
    x = np.arange(len(bandas))
    w = 0.26
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, titulo in zip(ejes, ["sesgo_medio_usd", "mae_usd"],
                               ["Sesgo medio (predicho − observado)", "Error absoluto medio (MAE)"]):
        for k, m in enumerate(modelos):
            v = s[s["modelo"] == m].set_index("segmento").loc[bandas, col]
            ax.bar(x + (k - 1) * w, v, width=w - 0.03, color=COLOR[m], label=NOMBRES[m])
        n = s[s["modelo"] == "rf"].set_index("segmento").loc[bandas, "n"]
        etiquetas = [f"{ETIQUETA_BANDA.get(b, b)}\nn={v:,}".replace(",", ".") for b, v in zip(bandas, n)]
        ax.set_xticks(x, etiquetas, fontsize=8.5)
        ax.axhline(0, color=TINTA_2, lw=0.8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
        ax.grid(axis="y", color=REJILLA, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(titulo, loc="left")
        ax.set_ylabel("Miles de USD de 1990")
        ax.set_xlabel("Banda de valor observado (cientos de miles de USD de 1990)")
    ejes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Error por banda de valor observado, diseño espacial", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=TINTA, y=1.04)
    fig.text(0.01, 0.955, "Los modelos sobreestiman las áreas baratas y subestiman las caras. En el techo, el error se "
             "mide contra 5,00001 (500.001 USD), no contra el valor real.", fontsize=9, color=TINTA_2)
    fig.savefig(destino / "fig3_error_por_banda.png")
    plt.close(fig)


def fig_ecdf(pred, cfg, destino):
    usd = cfg["datos"]["unidad_objetivo_usd"]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, d in zip(ejes, ["aleatorio_5f", "espacial_5f"]):
        sub = pred[pred["diseno"] == d]
        for m in ["mco", "gb", "rf"]:
            e = np.sort(np.abs(sub[f"pred_{m}"] - sub["observado"]).to_numpy() * usd)
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=COLOR[m], lw=2, label=NOMBRES[m])
            p90 = np.percentile(e, 90)
            ax.plot([p90], [0.9], "o", color=COLOR[m], ms=6, mec="white", mew=1.5)
        ax.axhline(0.9, color=REJILLA, lw=1, zorder=0)
        ax.set_xlim(0, 200000)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(color=REJILLA, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(DISENOS[d], loc="left")
        ax.set_xlabel("Error absoluto, miles de USD de 1990")
    ejes[0].set_ylabel("Proporción de block groups con error ≤ x")
    ejes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Distribución acumulada del error absoluto fuera de muestra", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=TINTA, y=1.04)
    fig.text(0.01, 0.955, "Punto: percentil 90 del error absoluto de cada modelo.", fontsize=9, color=TINTA_2)
    fig.savefig(destino / "fig4_distribucion_error.png")
    plt.close(fig)


def generar(met, seg, cel, pred, cfg, destino):
    fig_rmse(met, cfg, destino)
    fig_mapa(cel, cfg, destino)
    fig_bandas(seg, cfg, destino)
    fig_ecdf(pred, cfg, destino)
