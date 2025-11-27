import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from matplotlib.lines import Line2D  # Necessário para a legenda customizada

# --- 1. RAW DATA DEFINITION (Strategies and Articles) ---
# (Mantive os mesmos dados do seu exemplo anterior)
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

# Processamento dos dados (igual ao anterior)
expanded_data = []
year_regex = re.compile(r"(\d{4})$")

for category, citations_str in raw_data:
    citations_str = (
        citations_str.replace(r"\cite{", "")
        .replace("}", "")
        .replace(r", \colorbox{yellow}{", "")
        .replace(r"}", "")
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

# --- 2. PLOTAGEM COM LEGENDA GLOBAL E ESCALA REDUZIDA ---

categories = trend_df['Category'].unique()
num_categories = len(categories)

# Definição do Limite Fixo (conforme solicitado)
y_limit = 7 

# Layout
n_cols = 5
n_rows = (num_categories + n_cols - 1) // n_cols

# Ajuste do tamanho da figura (reduzi um pouco a altura já que o eixo Y é menor)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharex=True, sharey=True)
axes = axes.flatten()

all_years = sorted(trend_df['Year'].unique())
min_year = min(all_years)
max_year = max(all_years)
year_range = np.arange(min_year, max_year + 1, 1)

for i, category in enumerate(categories):
    ax = axes[i]
    category_data = trend_df[trend_df['Category'] == category]

    full_series = pd.Series(0, index=year_range)
    for year, count in zip(category_data['Year'], category_data['Article Count']):
        if year in year_range:
            full_series[year] = count

    # Plot dos Artigos
    # Mantivemos a cor padrão do pandas/matplotlib (geralmente azul)
    full_series.plot(kind='line', ax=ax, marker='o', label='_nolegend_')

    # Cálculo e Plot da Linha de Tendência
    x = full_series[full_series > 0].index.values
    y = full_series[full_series > 0].values

    if len(x) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        trend_line = intercept + slope * year_range
        ax.plot(year_range, trend_line, color='red', linestyle='--', linewidth=1, label='_nolegend_')

    # Formatação
    ax.set_title(category, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Eixo X: Todos os anos + rotação
    ax.set_xticks(year_range)
    ax.set_xlim(min_year - 0.5, max_year + 0.5)
    ax.tick_params(axis='x', rotation=45)
    
    # Eixo Y: Limite Fixo em 7
    ax.set_ylim(0, y_limit)
    
    # Remove labels individuais
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # NÃO adicionamos ax.legend() aqui dentro

# Ocultar subplots vazios
for i in range(num_categories, n_rows * n_cols):
    fig.delaxes(axes[i])

# --- 3. CRIAR LEGENDA GLOBAL ---
# Criamos elementos manuais para a legenda, garantindo que apareça correto mesmo que o último gráfico não tenha tendência
legend_elements = [
    Line2D([0], [0], color='#1f77b4', marker='o', label='Articles per Year'), # Azul padrão
    Line2D([0], [0], color='red', lw=1, linestyle='--', label='Trend Line')
]

# Posiciona a legenda no topo da figura, centralizada
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.98))

# Rótulos Globais dos Eixos
fig.supxlabel('Year', fontsize=14, y=0.01)
fig.supylabel('Number of Articles', fontsize=14, x=0.01)

# Ajustes de Layout (Tight Layout + Ajuste manual para dar espaço à legenda no topo e labels em baixo)
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.06) 

plt.savefig('trends_final_v3.png')
plt.show()
