import pandas as pd

# read dataset
df = pd.read_csv("mushroom.csv")

# change labels
df["class"] = df["class"].map({"e": "edible", "p": "poisonous"})

# ================================
# mapping value -> full name
# ================================

mapping = {
"cap-shape":{
"b":"bell","c":"conical","x":"convex","f":"flat","k":"knobbed","s":"sunken"
},
"cap-surface":{
"f":"fibrous","g":"grooves","y":"scaly","s":"smooth"
},
"cap-color":{
"n":"brown","b":"buff","c":"cinnamon","g":"gray","r":"green","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"
},
"bruises":{
"t":"bruises","f":"no"
},
"odor":{
"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"
},
"gill-attachment":{
"a":"attached","d":"descending","f":"free","n":"notched"
},
"gill-spacing":{
"c":"close","w":"crowded","d":"distant"
},
"gill-size":{
"b":"broad","n":"narrow"
},
"gill-color":{
"k":"black","n":"brown","b":"buff","h":"chocolate","g":"gray","r":"green","o":"orange","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"
},
"stalk-shape":{
"e":"enlarging","t":"tapering"
},
"stalk-root":{
"b":"bulbous","c":"club","u":"cup","e":"equal","z":"rhizomorphs","r":"rooted","?":"missing"
},
"stalk-surface-above-ring":{
"f":"fibrous","y":"scaly","k":"silky","s":"smooth"
},
"stalk-surface-below-ring":{
"f":"fibrous","y":"scaly","k":"silky","s":"smooth"
},
"stalk-color-above-ring":{
"n":"brown","b":"buff","c":"cinnamon","g":"gray","o":"orange","p":"pink","e":"red","w":"white","y":"yellow"
},
"stalk-color-below-ring":{
"n":"brown","b":"buff","c":"cinnamon","g":"gray","o":"orange","p":"pink","e":"red","w":"white","y":"yellow"
},
"veil-type":{
"p":"partial","u":"universal"
},
"veil-color":{
"n":"brown","o":"orange","w":"white","y":"yellow"
},
"ring-number":{
"n":"none","o":"one","t":"two"
},
"ring-type":{
"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"
},
"spore-print-color":{
"k":"black","n":"brown","b":"buff","h":"chocolate","r":"green","o":"orange","u":"purple","w":"white","y":"yellow"
},
"population":{
"a":"abundant","c":"clustered","n":"numerous","s":"scattered","v":"several","y":"solitary"
},
"habitat":{
"g":"grasses","l":"leaves","m":"meadows","p":"paths","u":"urban","w":"waste","d":"woods"
}
}

# =================================
# compute probability
# =================================

rows = []

for col in df.columns:

    if col == "class":
        continue

    table = pd.crosstab(df[col], df["class"], normalize="index")

    for value in table.index:

        prob_poison = table.loc[value,"poisonous"]

        full_value = mapping[col].get(value,value)

        rows.append({
            "feature":col,
            "value":full_value,
            "poisonous_probability":prob_poison
        })

result = pd.DataFrame(rows)

# sort 
result = result.sort_values("poisonous_probability",ascending=False)

# ranking
result.insert(0,"rank",range(1,len(result)+1))

# convert to %
result["poisonous_probability"] = (result["poisonous_probability"]*100).round(2)

# save file
result.to_csv("mushroom_poisonous_probability_ranked.csv",index=False)

print("File saved: mushroom_poisonous_probability_ranked.csv")
