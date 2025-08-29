import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dados corrigidos
data = {
    "Strategies": [
        "use of ontologies", "common metadata schema", "Rule-Based",
        "Metadata-driven framework", "Linked Data resources",
        "AI-based approaches", "Controlled vocabularies", "Annotations",
        "Schema matching", "String similarity", "Cardinality comparison",
        "Synonym detection", "Crowdsourcing", "Type alignment",
        "Domain and range constraints", "Parent-child relationships",
        "Path-based similarity", "Unit normalization",
        "Graph-oriented Models", "Data value comparison"
    ],
    "ISL": [4, 2, 4, 3, 4, 3, 1, 0, 0, 1, 2, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    "SCIoT": [3, 3, 4, 4, 3, 3, 2, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "GE": [6, 4, 2, 4, 1, 2, 3, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    "CDH": [7, 7, 6, 4, 3, 7, 4, 3, 3, 3, 2, 3, 3, 2, 1, 2, 2, 0, 0, 0],
    "DS": [9, 1, 5, 4, 5, 1, 1, 1, 2, 1, 0, 1, 0, 1, 0, 2, 0, 2, 1, 1],
    "HM": [14, 11, 3, 3, 3, 2, 6, 6, 3, 3, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0]
}

# Criar DataFrame
df = pd.DataFrame(data)
df.set_index("Strategies", inplace=True)

# Criar heatmap
plt.figure(figsize=(14, 10))
ax = sns.heatmap(df, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={'label': 'Frequency'})

# Colocar os nomes dos domínios no topo
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.set_xlabel("Domains", fontsize=12, weight="bold")

plt.ylabel("Strategies, Methods and Techniques")
plt.tight_layout()
plt.show()
