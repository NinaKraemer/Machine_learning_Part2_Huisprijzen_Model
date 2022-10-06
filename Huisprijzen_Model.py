# data inlezen en beschrijven
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from displayfunction import display

#%matplotlib inline

df = pd.read_json("nl_for_sale_all_anon.json")

df["vraagprijs"] = pd.to_numeric(df["vraagprijs"].str.replace(".",""))
df["bouwjaar"] = pd.to_numeric(df["bouwjaar"].str.replace("\w+\s+","", regex=True))
df["perceel"] = pd.to_numeric(df["perceel"])
df.dropna(subset=['vraagprijs']) # we kijken alleen naar huizen waar de we vraagprijs van weten
df.info()

#features kijken
y = df['vraagprijs']
X = df[['woonoppervlakte','slaapkamers','crawled_in','vraagprijsm2','perceel','bouwjaar']] # df[['feature1','feature2']] ...


#Zoals we bij het verkennen van de data gezien hebben komen er in de data huizen voor
# waarbij niet elke feature een waarde heeft. Om de data geschikt te maken om hier op een model
# op te trainen moeten deze waardes opgevuld worden, voor het gemak vullen we al deze waardes nu met 0 op
X = X.fillna(0)
X

#Splitsen train/test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


plt.scatter(X['woonoppervlakte'], y)
plt.show()


#Trainen
from sklearn.linear_model import LinearRegression
#Eerst maken we ons model aan
lr = LinearRegression()

#Hierna trainen we het model op de trainingsdata om de coefficienten voor ons model te berekenen
lr.fit(X_train, y_train)

#Evaluatie
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def print_results(data, actual, predictions):
  print("vraagprijs = " , lr.intercept_ , " + " , " + ".join(["{} * {}".format(el[0],el[1]) for el in list(zip(lr.coef_, X_train.columns))]))
  print("MSR: ", mean_squared_error(actual,predictions)**(1/2))
  print("R2", r2_score(actual,predictions))

  results = data.copy()
  results['predicted'] = predictions
  results['true'] = y_test
  results['difference'] = results['predicted'] - results['true']

  display(results.iloc[results['difference'].abs().argsort()])

predictions = lr.predict(X_test)
print_results(X_test, y_test, predictions)
