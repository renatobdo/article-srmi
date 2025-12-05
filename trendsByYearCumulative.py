import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D  # Para a legenda customizada

# --- 1. RAW DATA DEFINITION (Strategies and Articles) ---
raw_data = [
    ("Use of Ontologies", r"\cite{Guo2017, Novytskyi2025, Patrcio2025, Thalhath2025, Zeginis2024, An2020, Singh2022, Boldrini2024, Masmoudi2019, Ziaimatin2020, Pussi2025, Cavallo2019, Cheung2015, Candela2019, kimdois2015, Lisena2017, Ranjgar2024, Buranarach2022, Seifert2017, Chortaras2018, Alserafi2016, Cverdelj-Fogarai2017, Debruyne2015, Vysotska2024, IrhamFebrieka2025, Hu2024, Sawarkar2021, Do2015, Ashish2016, Rashid2020, Bachir2025, Griffiths2017, Kobayashi2018, Tian2025, Nicholson2020, Dugas2016, Sasse2022, Naji2022, Anguita2015, Hu2014, Peng2024}, \colorbox{yellow}{\cite{He2018}}"),
    ("MDLSP", r"\cite{Thalhath2025, Stiller2014, Verstockt2018, Alvarez2022, An2020, Koh2018, Ma2017, Bailo2023, Boldrini2024, Cavallo2019, Cheung2015, Candela2019, Buranarach2022, Seifert2017, Sivakumar2014, Alserafi2016, Dodois2015, Vysotska2024, Prabhune2018, Kandogan2015, Wang2014, Kobayashi2018, Lafia2018, Giuliacci2025, Bachir2025, Pussi2025, Gyrard2025, Chortaras2018, Qu2014}"),
    ("Common Metadata Schema", r"\cite{Chen2015, Novytskyi2025, Zeginis2024, Koh2018, Ma2017, Boldrini2024, Ziaimatin2020, Pussi2025, Horsburgh2014, kimdois2015, Lisena2017, Orgel2015, Buranarach2022, Mannocci2014, Seifert2017, Chortaras2018, Hu2024, Ashish2016, Kandogan2015, Leroux2017, Rashid2020, Kobayashi2018, Nicholson2020, Naji2022, Hegselmann2021, Sangkla2017, Simon2024, Gyrard2025, Demraoui2016}"),
    ("Rule-Based Approaches", r"\cite{Chen2015, Patrcio2025, Thalhath2025, Li2025, Zeginis2024, Koh2018, Singh2022, Ma2017, Masmoudi2019, Cheung2015, Candela2019, kimdois2015, Lisena2017, Orgel2015, Ranjgar2024, Mannocci2014, Cverdelj-Fogarai2017, Dodois2015, Joonas2025, Sawarkar2021, Do2015, Kobayashi2018, Naji2022, Peng2024, Qu2014, Rashid2020, Kirsten2017}"),
    ("Schema Matching Approaches", r"\cite{Giuliacci2025, Orgel2015, Buranarach2022, Mannocci2014, Alserafi2016, IrhamFebrieka2025, Ashish2016, Iancu2024, Anguita2015, McCrae2015, Kirsten2017, Ma2017, Chen2015, Thalhath2025, Hu2014, Cheung2015, kimdois2015, Zeginis2024, Dodois2015, Joonas2025}"),
    ("Linked Data Resources", r"\cite{Novytskyi2025, McCrae2015, Thalhath2025, DeSantis2025, Zeginis2024, An2020, Singh2022, Hu2015, Candela2019, Seifert2017, Kaldeli2024, Debruyne2015, Dodois2015, Idrissou2017, Khalid2018, Do2015, Kobayashi2018, Nicholson2020, Hu2014, Alvarez2022}"),
    ("Controlled Vocabularies", r"\cite{McCrae2015, Zeginis2024, Lafia2018, Giuliacci2025, Horsburgh2014, Wang2016, Lisena2017, Ranjgar2024, Mannocci2014, Gil2023, Joonas2025, Kirsten2017, Rashid2020, Kobayashi2018, Sasse2022, Anguita2015, kimdois2015, Buranarach2022, Sawarkar2021}"),
    ("AI-Based Approaches", r"\cite{McCrae2015, DeSantis2025, Verstockt2018, Zeginis2024, Lafia2018, Ma2017, Cheung2015, Hu2015, Candela2019, Ranjgar2024, Seifert2017, Chortaras2018, Kaldeli2024, Kaldeli2021, Sawarkar2021, Ashish2016, Cannizzaro2021}"),
    ("Annotations", r"\cite{Verstockt2018, An2020, Lafia2018, Cavallo2019, Chortaras2018, Kaldeli2024, Kaldeli2021, Sawarkar2021, Kirsten2017, Rashid2020, Dugas2016, Sasse2022, Ulrich2022}"),
    ("Graph-Oriented Models", r"\cite{Bunakov2019, Lisena2017, Ma2017}"),
]

# --- 2. PROCESSAMENTO DOS DADOS ---

expanded_data = []
year_regex = re.compile(r"(\d{4})$")

for category, citations_str in raw_data:
    citations_str = (
        citations_str.replace(r"\cite{", "")
        .replace("}", "")
        .replace(r", \colorbox{yellow}{", "")
        .replace(r"\colorbox{yellow}{", "")
    )
    citation_keys = [key.strip() for key in citations_str.split(',')]
    for key in citation_keys:
        match = year_regex.search(key)
        if match:
            year = int(match.group(1))
            expanded_data.append({"Category": category, "Year": year})

df = pd.DataFrame(expanded_data)
trend_df = df.groupby(['Category', 'Year']).size().reset_index(name='Article Count')

# --- 3. DEFININDO ANOS E LIMITE DO EIXO Y (ACUMULADO) ---

categories = trend_df['Category'].unique()
num_categories = len(categories)

all_years = sorted(trend_df['Year'].unique())
min_year = min(all_years)
max_year = max(all_years)
year_range = np.arange(min_year, max_year + 1, 1, dtype=int)

# Máximo acumulado por categoria (para definir limite do eixo Y)
max_total_per_category = trend_df.groupby('Category')['Article Count'].sum().max()
y_limit = max_total_per_category + 1  # folga visual

# --- 4. LAYOUT DOS SUBPLOTS ---

n_cols = 5
n_rows = (num_categories + n_cols - 1) // n_cols

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(18, 4 * n_rows),
    sharex=True,
    sharey=True
)
axes = axes.flatten()

# --- 5. LOOP POR CATEGORIA (SÉRIE ACUMULADA) ---

for i, category in enumerate(categories):
    ax = axes[i]
    category_data = trend_df[trend_df['Category'] == category]

    # Série com contagem por ano (inclui zeros)
    full_series = pd.Series(0, index=year_range)
    for year, count in zip(category_data['Year'], category_data['Article Count']):
        if year in year_range:
            full_series[year] = count

    # Série acumulada
    cumulative_series = full_series.cumsum()

    # Plot da série acumulada
    cumulative_series.plot(
        kind='line',
        ax=ax,
        marker='o',
        label='_nolegend_'
    )

    # --- TICKS NO EIXO X ---

    # major ticks: de 2 em 2 anos (2014, 2016, 2018, ...), garantindo o último ano
    major_ticks = list(year_range[::2])
    if year_range[-1] not in major_ticks:
        major_ticks.append(year_range[-1])

    ax.set_xticks(major_ticks)                  # major ticks com rótulo
    ax.set_xticks(year_range, minor=True)       # minor ticks em TODOS os anos

    # estilo dos ticks
    ax.tick_params(axis='x', which='major', length=6, width=1)
    ax.tick_params(axis='x', which='minor', length=3, width=0.8)

    # rótulos só nos major ticks
    ax.set_xticklabels([str(t) for t in major_ticks], rotation=45, ha='right')

    # limites do eixo X exatamente nos anos extremos
    ax.set_xlim(year_range[0], year_range[-1])

    # Formatação do subplot
    ax.set_title(category, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, y_limit)

    ax.set_xlabel('')
    ax.set_ylabel('')

# Esconde subplots vazios (se existirem)
for i in range(num_categories, n_rows * n_cols):
    fig.delaxes(axes[i])

# --- 6. LEGENDA GLOBAL ---

legend_elements = [
    Line2D(
        [0], [0],
        color='#1f77b4',
        marker='o',
        label='Cumulative number of articles'
    )
]

fig.legend(
    handles=legend_elements,
    loc='upper center',
    ncol=1,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.98)
)

# --- 7. RÓTULOS GLOBAIS DOS EIXOS ---

fig.supxlabel('Year', fontsize=14, y=0.01)
fig.supylabel('Cumulative number of articles', fontsize=14, x=0.01)

# --- 8. AJUSTES FINAIS ---

plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.06)

plt.savefig('trends_cumulative_ticks_ok.png', dpi=300, bbox_inches='tight')
plt.show()
