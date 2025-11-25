import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# === Dataset (mesmos blocos) ===
blocks = {
    "Ontologies": """
        Guo2017, Novytskyi2025, Patrcio2025, Thalhath2025, Zeginis2024, An2020, 
        Singh2022, Boldrini2024, Masmoudi2019, Ziaimatin2020, Pussi2025, 
        Cavallo2019, Cheung2015, Candela2019, Kimdois2015, Lisena2017, 
        Ranjgar2024, Buranarach2022, Seifert2017, Chortaras2018, 
        Alserafi2016, Cverdelj-Fogarai2017, Debruyne2015, Vysotska2024, 
        IrhamFebrieka2025, Hu2024, Sawarkar2021, Do2015, Ashish2016, 
        Rashid2020, Bachir2025, Griffiths2017, Kobayashi2018, Tian2025, 
        Nicholson2020, Dugas2016, Sasse2022, Naji2022, Anguita2015, Hu2014, 
        Peng2024, He2018
    """,

    "Metadata standards": """
        Thalhath2025, Stiller2014, Verstockt2018, Alvarez2022, An2020, 
        Koh2018, Ma2017, Bailo2023, Boldrini2024, Cavallo2019, Cheung2015, 
        Candela2019, Buranarach2022, Seifert2017, Sivakumar2014, 
        Alserafi2016, dodois2015, Vysotska2024, Prabhune2018, Kandogan2015, 
        Wang2014, Kobayashi2018, Lafia2018, Giuliacci2025, Bachir2025, 
        Pussi2025, Gyrard2025, Chortaras2018, Qu2014
    """,

    "Common metadata schema": """
        Chen2015, Novytskyi2025, Zeginis2024, Koh2018, Ma2017, Boldrini2024, 
        Ziaimatin2020, Pussi2025, Horsburgh2014, kimdois2015, Lisena2017, 
        Orgel2015, Buranarach2022, Mannocci2014, Seifert2017, Chortaras2018,
        Hu2024, Ashish2016, Kandogan2015, Leroux2017, Rashid2020, 
        Kobayashi2018, Nicholson2020, Naji2022, Hegselmann2021,
        Sangkla2017, Simon2024, Gyrard2025, Demraoui2016
    """,

    "Rule-based": """
        Chen2015, Patrcio2025, Thalhath2025, Li2025, Zeginis2024, Koh2018, 
        Singh2022, Ma2017, Masmoudi2019, Cheung2015, Candela2019, 
        kimdois2015, Lisena2017, Orgel2015, Ranjgar2024, Mannocci2014, 
        Cverdelj-Fogarai2017, dodois2015, Joonas2025, Sawarkar2021, 
        Do2015, Kobayashi2018, Naji2022, Peng2024, Qu2014, Rashid2020,
         Kirsten2017
    """,

    "Schema matching": """
        Giuliacci2025, Orgel2015, Buranarach2022, Mannocci2014, Alserafi2016, 
        IrhamFebrieka2025, Ashish2016, Iancu2024, Anguita2015, McCrae2015, 
        Kirsten2017, Ma2017, Chen2015, Thalhath2025, Hu2014, Cheung2015,  
        kimdois2015, Zeginis2024, dodois2015, Joonas2025
    """,

    "Linked Data": """
       Novytskyi2025, McCrae2015, Thalhath2025, DeSantis2025, Zeginis2024, 
       An2020, Singh2022, Hu2015, Candela2019, Seifert2017, Kaldeli2024, 
       Debruyne2015, dodois2015, Idrissou2017, Khalid2018, Do2015, Kobayashi2018, 
       Nicholson2020, Hu2014, Alvarez2022
    """,

    "Controlled vocabularies": """
        McCrae2015, Zeginis2024, Lafia2018, Giuliacci2025, Horsburgh2014, 
        Wang2016, Lisena2017, Ranjgar2024, Mannocci2014, Gil2023, Joonas2025,
        Kirsten2017, Rashid2020, Kobayashi2018, Sasse2022, Anguita2015, 
        kimdois2015, Buranarach2022, Sawarkar2021
    """,

    "AI-based": """
        McCrae2015, DeSantis2025, Verstockt2018, Zeginis2024, Lafia2018, 
        Ma2017, Cheung2015, Hu2015, Candela2019, Ranjgar2024, Seifert2017, 
        Chortaras2018, Kaldeli2024, Kaldeli2021, Sawarkar2021, Ashish2016, 
        Cannizzaro2021
    """,

    "Annotations": """
        Verstockt2018, An2020, Lafia2018, Cavallo2019, Chortaras2018, 
        Kaldeli2024, Kaldeli2021, Sawarkar2021, Kirsten2017, Rashid2020, 
        Dugas2016, Sasse2022, Ulrich2022
    """,

    "Graph models": """
       Bunakov2019, Lisena2017, Ma2017
    """
}

def extract_years(txt):
    years = re.findall(r"(\d{4})", txt)
    return [int(y) for y in years if 2010 < int(y) < 2030]

# Abreviações das categorias
abbr = {
    "Ontologies": "on",
    "Common metadata schema": "cm",
    "Metadata standards": "md",   # metadata-driven language, standards and processes
    "Rule-based": "rb",
    "Schema matching": "sm",
    "Linked Data": "ld",
    "Controlled vocabularies": "cv",
    "AI-based": "ai",
    "Annotations": "an",
    "Graph models": "gm"
}

# Contagens por ano e categoria
years_all = set()
counts = {}
totals = {}

for cat, txt in blocks.items():
    ys = extract_years(txt)
    c = Counter(ys)
    counts[cat] = c
    totals[cat] = sum(c.values())
    years_all.update(c.keys())

years_all = sorted(years_all)

# Ordenar categorias da mais frequente para a menos
sorted_cats = sorted(totals.keys(), key=lambda c: totals[c], reverse=True)

# Paleta forte amarelo → laranja → vermelho
palette = [
    "#fff700",  # bright yellow
    "#ffe200",
    "#ffcc00",
    "#ffb000",
    "#ff9500",
    "#ff7a00",
    "#f25f00",
    "#d94a00",
    "#b23500",
    "#7f1f00"   # dark red
]
color_map = {cat: palette[i] for i, cat in enumerate(sorted_cats)}

# Plot
plt.figure(figsize=(16, 8))
bottom = np.zeros(len(years_all))

for cat in sorted_cats:
    vals = np.array([counts[cat].get(y, 0) for y in years_all])
    plt.bar(
        years_all,
        vals,
        bottom=bottom,
        color=color_map[cat],
        label=f"{cat} ({abbr[cat]})"
    )

    # adiciona sigla dentro das barras (onde há valor > 0)
    for i, (year, v, b) in enumerate(zip(years_all, vals, bottom)):
        if v > 0:
            y_center = b + v / 2.0
            plt.text(
                year,
                y_center,
                abbr[cat],
                ha="center",
                va="center",
                fontsize=7,
                color="black"
            )

    bottom += vals

plt.xlabel("Year")
plt.ylabel("Number of Articles")
plt.title("Annual Distribution by Category (with Abbreviations in Bars)")
plt.yticks(range(0, int(max(bottom)) + 1, 1))
plt.grid(True, axis='y', linestyle='--', alpha=0.35)
plt.xticks(years_all, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Category")

plt.tight_layout()
plt.show()
