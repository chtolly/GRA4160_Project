import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("mushroom.csv")

df["class"] = df["class"].map({"e": "edible", "p": "poisonous"})

data = []

# compute % poisonous for each feature value
for col in df.columns:
    
    if col == "class":
        continue
        
    table = pd.crosstab(df[col], df["class"], normalize="index") * 100
    
    for value in table.index:
        data.append({
            "feature": col,
            "value": value,
            "poisonous_percent": table.loc[value, "poisonous"]
        })

heatmap_df = pd.DataFrame(data)

# pivot to draw heatmap
pivot = heatmap_df.pivot(index="value", columns="feature", values="poisonous_percent")

plt.figure(figsize=(14,8))
sns.heatmap(pivot, cmap="coolwarm", annot=False)

plt.title("Poisonous Percentage by Feature Value")
plt.ylabel("Feature Value")
plt.xlabel("Feature")
plt.show()
