# -*- coding: utf-8 -*-
# =========================================================
# Visualizações acessíveis (colorblind-safe) — ordenadas do maior para o menor
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 1. Dados originais
# ----------------------------
data = {
    "Strategies": [
        "use of ontologies", "common metadata schema", "Rule-Based",
        "MDLSP", "Schema matching approaches",
        "Linked Data resources", "Controlled vocabularies", "AI-based approaches",
        "Annotations", "Graph-oriented Models"
    ],
    "ISL": [4, 2, 4, 3, 3, 4, 1, 3, 1, 1],
    "SCIoT": [3, 3, 4, 6, 2, 4, 2, 3, 2, 1],
    "GE": [6, 4, 2, 4, 2, 1, 3, 2, 1, 0],
    "CDH": [7, 7, 7, 6, 4, 3, 6, 6, 3, 1],
    "DS": [8, 3, 6, 4, 4, 5, 2, 1, 1, 0],
    "HM": [14, 11, 5, 5, 5, 3, 5, 2, 5, 0]
}

df = pd.DataFrame(data).set_index("Strategies")

# ----------------------------
# 2. Ordenar do maior total para o menor
# ----------------------------
df["Total"] = df.sum(axis=1)
df = df.sort_values("Total", ascending=False)
df.drop(columns="Total", inplace=True)

# Guardar a ordem
order_strat = list(df.index)
order_domain = list(df.columns)

# ----------------------------
# 3. Configuração de estilo
# ----------------------------
sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)
CB_PALETTE = sns.color_palette("tab10")
POINT_PALETTE = sns.color_palette("Set2")
BAR_EDGE = "black"

# ----------------------------
# 4. Formato longo
# ----------------------------
df_long = df.reset_index().melt(id_vars="Strategies", var_name="Domain", value_name="Frequency")
df_long["Strategies"] = pd.Categorical(df_long["Strategies"], categories=order_strat, ordered=True)
df_long["Domain"] = pd.Categorical(df_long["Domain"], categories=order_domain, ordered=True)

# =========================================================
# A) Grouped Bar Chart
# =========================================================
plt.figure(figsize=(14, 8))
ax = sns.barplot(
    data=df_long, x="Frequency", y="Strategies", hue="Domain",
    palette=CB_PALETTE, edgecolor=BAR_EDGE, order=order_strat
)
ax.set_title("Frequency of Strategies per Domain — Grouped Bar Chart", fontsize=14, weight="bold")
ax.set_xlabel("Frequency")
ax.set_ylabel("Strategies (ordered by total frequency)")
ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =========================================================
# B) Dot Plot
# =========================================================
plt.figure(figsize=(14, 8))
ax = sns.scatterplot(
    data=df_long, x="Frequency", y="Strategies", hue="Domain",
    palette=POINT_PALETTE, s=150, edgecolor="black"
)
ax.set_title("Frequency of Strategies across Domains — Dot Plot", fontsize=14, weight="bold")
ax.set_xlabel("Frequency")
ax.set_ylabel("Strategies (ordered by total frequency)")
ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =========================================================
# C) Lollipop Chart
# =========================================================
plt.figure(figsize=(14, 8))
for (_, row) in df_long.iterrows():
    plt.plot([0, row["Frequency"]], [row["Strategies"], row["Strategies"]], color="gray", alpha=0.6)
ax = sns.scatterplot(
    data=df_long, x="Frequency", y="Strategies", hue="Domain",
    palette=CB_PALETTE, s=150, edgecolor="black"
)
ax.set_title("Frequency of Strategies across Domains — Lollipop Chart", fontsize=14, weight="bold")
ax.set_xlabel("Frequency")
ax.set_ylabel("Strategies (ordered by total frequency)")
ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# =========================================================
# D) Bubble Map (uma cor) — com valores dentro das bolhas
# =========================================================
import matplotlib.patheffects as pe

plt.figure(figsize=(14, 8))

scale = 80  # fator de escala do tamanho da bolha
min_size = 24  # tamanho mínimo p/ bolhas > 0 (evita sumir)
bubble_color = "#4477AA"

for i, strategy in enumerate(order_strat):
    for j, domain in enumerate(order_domain):
        val = int(df.loc[strategy, domain])
        if val > 0:
            size = max(min_size, val * scale)
            plt.scatter(j, i, s=size, color=bubble_color, alpha=0.85, edgecolors="black")

            # Texto centralizado com contorno preto (melhor legibilidade)
            # Tamanho da fonte cresce levemente com o valor
            fs = 8 + 0.5 * min(val, 8)   # cap no crescimento para não ficar exagerado
            txt = plt.text(
                j, i, f"{val}",
                ha="center", va="center",
                fontsize=fs, color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")]
            )
        else:
            # Se quiser mostrar zeros como bolhinhas mínimas e texto "0", descomente:
            # plt.scatter(j, i, s=min_size, color=bubble_color, alpha=0.3, edgecolors="black")
            # plt.text(j, i, "0", ha="center", va="center", fontsize=8, color="white",
            #          path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            pass

plt.xticks(range(len(order_domain)), order_domain)
plt.yticks(range(len(order_strat)), order_strat)
plt.title("Bubble Map of Strategy Frequencies (ordered) — values inside bubbles", fontsize=14, weight="bold")
plt.xlabel("Domains")
plt.ylabel("Strategies (ordered by total frequency)")

# grade e layout
plt.gca().invert_yaxis()
plt.grid(True, axis="x", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.show()



# =========================================================
# E) Small Multiples — painéis ordenados por total do domínio (maior→menor),
#     mesma escala no eixo X, barras em ordem local e rótulos numéricos
# =========================================================
import matplotlib.ticker as mticker

NEUTRAL_COLOR = "#4B5563"  # tom neutro legível
global_rank = {s: i for i, s in enumerate(order_strat)}
df_long["GlobalRank"] = df_long["Strategies"].map(global_rank)

# 1) Ordenar DOMÍNIOS pelo total (maior→menor)
domain_totals = df.sum(axis=0)  # soma por coluna (domínio)
domain_order_by_total = domain_totals.sort_values(ascending=False).index.tolist()

# 2) Escala comum no eixo X
global_max = int(df_long["Frequency"].max())
xlim_max = global_max * 1.18  # “respiro” para rótulos

def facet_bar_sorted_with_labels_same_scale(data, **kwargs):
    # Ordena localmente as estratégias dentro do domínio (maior→menor); desempate pela ordem global
    order_local = (data.sort_values(["Frequency", "GlobalRank"],
                                    ascending=[False, True])["Strategies"].tolist())

    d = data.copy()
    d["StrategiesOrdered"] = pd.Categorical(d["Strategies"], categories=order_local, ordered=True)

    ax = sns.barplot(
        data=d,
        x="Frequency", y="StrategiesOrdered",
        color=NEUTRAL_COLOR, edgecolor="black"
    )
    ax.set_xlim(0, xlim_max)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Rótulos de valor
    for container in ax.containers:
        try:
            ax.bar_label(container,
                         labels=[f"{int(v)}" for v in container.datavalues],
                         padding=3, fontsize=9)
        except Exception:
            for p in container:
                w = p.get_width()
                y = p.get_y() + p.get_height()/2
                ax.annotate(f"{int(w)}", xy=(w, y), xytext=(3, 0),
                            textcoords="offset points", va="center", fontsize=9)

    ax.set_ylabel("")

g = sns.FacetGrid(
    df_long,
    col="Domain",
    col_order=domain_order_by_total,  # <- domínios com maiores totais primeiro (topo/esquerda)
    col_wrap=3,
    sharex=True,
    height=4
)
g.map_dataframe(facet_bar_sorted_with_labels_same_scale)

g.set_titles("{col_name}")
g.set_axis_labels("Frequency", "")
plt.subplots_adjust(top=0.88)
g.fig.suptitle("Distribution of Strategies, Methods and Techniques by Domain", fontsize=16, weight="bold")
plt.show()
