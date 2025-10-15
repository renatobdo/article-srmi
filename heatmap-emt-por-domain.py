import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dados corrigidos
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
