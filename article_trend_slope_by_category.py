import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# --- 1. RAW DATA DEFINITION (Strategies and Articles) ---
raw_data = [
    (
        "Use of Ontologies",
        r"\cite{Guo2017, Novytskyi2025, Patrcio2025, Thalhath2025, Zeginis2024, An2020, Singh2022, Boldrini2024, Masmoudi2019, Ziaimatin2020, Pussi2025, Cavallo2019, Cheung2015, Candela2019, kimdois2015, Lisena2017, Ranjgar2024, Buranarach2022, Seifert2017, Chortaras2018, Alserafi2016, Cverdelj-Fogarai2017, Debruyne2015, Vysotska2024, IrhamFebrieka2025, Hu2024, Sawarkar2021, Do2015, Ashish2016, Rashid2020, Bachir2025, Griffiths2017, Kobayashi2018, Tian2025, Nicholson2020, Dugas2016, Sasse2022, Naji2022, Anguita2015, Hu2014, Peng2024}, \colorbox{yellow}{\cite{He2018}}"
    ),
    (
        "MDLSP",
        r"\cite{Thalhath2025, Stiller2014, Verstockt2018, Alvarez2022, An2020, Koh2018, Ma2017, Bailo2023, Boldrini2024, Cavallo2019, Cheung2015, Candela2019, Buranarach2022, Seifert2017, Sivakumar2014, Alserafi2016, dodois2015, Vysotska2024, Prabhune2018, Kandogan2015, Wang2014, Kobayashi2018, Lafia2018, Giuliacci2025, Bachir2025, Pussi2025, Gyrard2025, Chortaras2018, Qu2014}"
    ),
    (
        "Common Metadata Schema",
        r"\cite{Chen2015, Novytskyi2025, Zeginis2024, Koh2018, Ma2017, Boldrini2024, Ziaimatin2020, Pussi2025, Horsburgh2014, kimdois2015, Lisena2017, Orgel2015, Buranarach2022, Mannocci2014, Seifert2017, Chortaras2018, Hu2024, Ashish2016, Kandogan2015, Leroux2017, Rashid2020, Kobayashi2018, Nicholson2020, Naji2022, Hegselmann2021, Sangkla2017, Simon2024, Gyrard2025, Demraoui2016}"
    ),
    (
        "Rule-Based Approaches",
        r"\cite{Chen2015, Patrcio2025, Thalhath2025, Li2025, Zeginis2024, Koh2018, Singh2022, Ma2017, Masmoudi2019, Cheung2015, Candela2019, kimdois2015, Lisena2017, Orgel2015, Ranjgar2024, Mannocci2014, Cverdelj-Fogarai2017, dodois2015, Joonas2025, Sawarkar2021, Do2015, Kobayashi2018, Naji2022, Peng2024, Qu2014, Rashid2020, Kirsten2017}"
    ),
    (
        "Schema Matching Approaches",
        r"\cite{Giuliacci2025, Orgel2015, Buranarach2022, Mannocci2014, Alserafi2016, IrhamFebrieka2025, Ashish2016, Iancu2024, Anguita2015, McCrae2015, Kirsten2017, Ma2017, Chen2015, Thalhath2025, Hu2014, Cheung2015, kimdois2015, Zeginis2024, dodois2015, Joonas2025}"
    ),
    (
        "Linked Data Resources",
        r"\cite{Novytskyi2025, McCrae2015, Thalhath2025, DeSantis2025, Zeginis2024, An2020, Singh2022, Hu2015, Candela2019, Seifert2017, Kaldeli2024, Debruyne2015, dodois2015, Idrissou2017, Khalid2018, Do2015, Kobayashi2018, Nicholson2020, Hu2014, Alvarez2022}"
    ),
    (
        "Controlled Vocabularies",
        r"\cite{McCrae2015, Zeginis2024, Lafia2018, Giuliacci2025, Horsburgh2014, Wang2016, Lisena2017, Ranjgar2024, Mannocci2014, Gil2023, Joonas2025, Kirsten2017, Rashid2020, Kobayashi2018, Sasse2022, Anguita2015, kimdois2015, Buranarach2022, Sawarkar2021}"
    ),
    (
        "AI-Based Approaches",
        r"\cite{McCrae2015, DeSantis2025, Verstockt2018, Zeginis2024, Lafia2018, Ma2017, Cheung2015, Hu2015, Candela2019, Ranjgar2024, Seifert2017, Chortaras2018, Kaldeli2024, Kaldeli2021, Sawarkar2021, Ashish2016, Cannizzaro2021}"
    ),
    (
        "Annotations",
        r"\cite{Verstockt2018, An2020, Lafia2018, Cavallo2019, Chortaras2018, Kaldeli2024, Kaldeli2021, Sawarkar2021, Kirsten2017, Rashid2020, Dugas2016, Sasse2022, Ulrich2022}"
    ),
    (
        "Graph-Oriented Models",
        r"\cite{Bunakov2019, Lisena2017, Ma2017}"
    ),
]

# Data Processing
expanded_data = []
regex_year = re.compile(r"(\d{4})$")

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
        match = regex_year.search(key)
        if match:
            year = int(match.group(1))
            expanded_data.append({"Category": category, "Year": year})

df = pd.DataFrame(expanded_data)
trend_df = df.groupby(['Category', 'Year']).size().reset_index(name='Article Count')


# --- CALCULATE SLOPES FOR PLOTTING ---
categories = trend_df['Category'].unique()
slopes = {}
min_year = trend_df['Year'].min()
max_year = trend_df['Year'].max()
year_range = np.arange(min_year, max_year + 1, 1)

for category in categories:
    category_data = trend_df[trend_df['Category'] == category]
    x = category_data['Year'].values
    y = category_data['Article Count'].values
    
    # Filter for sufficient data points
    if len(x) >= 2:
        slope, _, _, _, _ = linregress(x, y)
        slopes[category] = slope
    else:
        slopes[category] = np.nan

# Create Slopes DataFrame for sorting and bar chart
slopes_df = pd.DataFrame(list(slopes.items()), columns=['Category', 'Slope']).sort_values(by='Slope', ascending=False).fillna(0)

# Create Heatmap DataFrame (ordered by slope)
heatmap_df = trend_df.pivot_table(index='Category', columns='Year', values='Article Count').fillna(0)
heatmap_df = heatmap_df.reindex(slopes_df['Category'])


# --- PLOT 1: SLOPE BAR CHART (TREND RATE) ---
plt.figure(figsize=(14, 7)) # Increased height for better label spacing
# Define colors: Blue for growth (Slope > 0), Red for decline (Slope < 0)
colors = ['skyblue' if s > 0 else 'salmon' for s in slopes_df['Slope']]

# Prepare labels with line breaks for the bar chart
# Replace spaces with newlines, but only after 'Approaches' or 'Resources' or 'Schema' or 'Language'
# Also handle "Metadata-Driven Language, Standards and Processes" specifically
formatted_labels_bar = []
for label in slopes_df['Category']:
    if "Rule-Based Approaches" in label:
        formatted_labels_bar.append("Rule-Based\nApproaches")
    elif "AI-Based Approaches" in label:
        formatted_labels_bar.append("AI-Based\nApproaches")
    elif "Schema Matching Approaches" in label:
        formatted_labels_bar.append("Schema Matching\nApproaches")
    elif "Common Metadata Schema" in label:
        formatted_labels_bar.append("Common Metadata\nSchema")
    elif "Linked Data Resources" in label:
        formatted_labels_bar.append("Linked Data\nResources")
    elif "Controlled Vocabularies" in label:
        formatted_labels_bar.append("Controlled\nVocabularies")
    elif "Graph-Oriented Models" in label:
        formatted_labels_bar.append("Graph-Oriented\nModels")
    else: # Default for "Use of Ontologies", "Annotations"
        formatted_labels_bar.append(label)

plt.barh(formatted_labels_bar, slopes_df['Slope'], color=colors)

plt.axvline(0, color='gray', linestyle='--') # Reference line at zero
plt.title('Article Trend (Slope) by Category', fontsize=16)
plt.xlabel('Linear Regression Slope', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.gca().invert_yaxis() # Put highest growth at the top
plt.tight_layout()
plt.show()
