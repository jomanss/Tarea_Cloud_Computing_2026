
#Librerias
import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("python_learning_exam_performance.csv")
print(df.head())

print("\Chequeamos las dimensiones:")
print(df.shape)

print("\nRevision de valores nulos")
print(df.isnull().sum() )

print("\nRevision de valores nulos")
print(df.isnull().sum()/len(df)*100) # Un tercio de la data es nula, por lo que se opta por eliminar

print()
print(df.nunique())

print("De la tabla se concluye que efectivamente student_id es un id(unico para cada registro) y se deberá eliminar")
print("Adicional se visualizan 3 variables dicotomicas entre ellas la objetivo: passed exam")

#De la tabla se concluye que efectivamente student_id es un id(unico para cada registro) y se deberá eliminar
#Adicional se visualizan 3 variables dicotomicas entre ellas la objetivo "passed exam"

print("\nRevision Descriptiva\n", df.describe(include= "all").T)
print()

###################Graficas Exploratorias##############################
var_num = [col for col in df.select_dtypes("number").columns if df[col].nunique()>2 and col != "student_id"]
def histplot_df(df, vars, target = None):
    sns.set_palette("Set2")
    fig, axes = plt.subplots(3,3, figsize = (10,10))
    axes = axes.flatten()
    

    for i, col in enumerate(var_num):
        axes[i].set_title(f" {col}")
        sns.histplot(df, x = col, ax = axes[i], bins = 15, hue = target)

    plt.tight_layout()

histplot_df(df, var_num)
plt.show()

#histograma en base a variables objetivo
histplot_df(df, var_num, "passed_exam")
plt.show()

#Matriz de correlacion

sns.heatmap(df[var_num].corr(), annot= True,
            fmt = ".2f",
            cmap = "coolwarm")
plt.title("Matriz de Correlacion")
plt.show()



#Modelado
def drop_specific_columns(x):
    drop_cols = ["prior_programming_experience", "student_id", "final_exam_score"]
    return x.drop(drop_cols, axis=1, errors="ignore")


drop_cols = ["prior_programming_experience", "student_id", "final_exam_score"] #Columnas a eliminar

num_var = make_column_selector(dtype_include="number")
cat_var = make_column_selector(dtype_include="object")

pipe_drop = FunctionTransformer(drop_specific_columns)

pipe_num = Pipeline(steps = [("imp_num", SimpleImputer(strategy = "median"))])

pipe_cat = Pipeline(steps = [("imp_cat", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder())])

pp = ColumnTransformer( transformers= [("num", pipe_num, num_var),
                                            ("cat", pipe_cat, cat_var)],
                                         remainder = "passthrough"
                                )

pipe_classifier = {
    
    "LogisticRegression" : {

        "model": LogisticRegression(),
        "params": {"classifier__C": [0.1,1,10],
                   "classifier__random_state" : [42],
                   "selector__k": [4, 6, 8],

                   }
    },

    "DecisionTree" : {

        "model" : DecisionTreeClassifier(),
        "params": {"classifier__max_depth": [1,2,3],
                   "classifier__criterion": ["gini"],
                   "classifier__random_state" : [42],
                   "selector__k": [4, 6, 8],
                   }

    },

    "RandomForest" : {

        "model": RandomForestClassifier(),
        "params": {"classifier__n_estimators": [100, 200,300],
                   "classifier__random_state" : [42],
                   "selector__k": [4, 6, 8],
                   }

    },

    "XGBoost" : {

        "model": XGBClassifier(),
        "params": {"classifier__n_estimators": [100, 200,300],
                   "classifier__learning_rate" : [0.01, 0.02],
                   "classifier__random_state" : [42],
                   "selector__k": [4, 6, 8],
                   }
    },

}

target = "passed_exam"

df_x = df.drop(target, axis = 1)
df_y = df[target]

x_train, x_test , y_train, y_test = train_test_split(df_x, df_y, test_size= 0.5, random_state= 7)

mejor_modelo = {}

print("Evaluacion de modelos:\n")
for name, modelo_data in pipe_classifier.items():
    pipe_final = Pipeline( steps = [
        ("drop", pipe_drop),
        ("preprocessor",  pp),
        ("selector", SelectKBest(score_func=f_classif)),
        ("classifier", modelo_data["model"])
    ])

    grid = GridSearchCV(pipe_final, modelo_data["params"], scoring = "accuracy", cv = 5)
    grid.fit(x_train, y_train)

    mejor_modelo[name] = {"mejor_accuracy" : grid.best_score_,
                          "mejores_parametros": grid.best_params_,
                          "mejor_estimator": grid.best_estimator_}
    
    print(f"{name} optimizado con Accuracy: {grid.best_score_:.4f}")

ganador = max(mejor_modelo, key=lambda x: mejor_modelo[x]['mejor_accuracy'])

print(f"\nEl mejor modelo es {ganador}")

#Impresion de Formula
best_pipe = mejor_modelo[ganador]["mejor_estimator"]

variables = best_pipe.named_steps["preprocessor"].get_feature_names_out()
filtro_var_selec = best_pipe.named_steps["selector"].get_support()

var_finales = variables[filtro_var_selec]

coefs = best_pipe.named_steps["classifier"].coef_.ravel()
intercepto = float(best_pipe.named_steps["classifier"].intercept_)

print(f"Las variables finales seleccionadas son: {var_finales}")

df_formula = pd.DataFrame({
    "feature": var_finales,
    "beta": coefs,
    "odds_ratio": np.exp(coefs)
}).sort_values("beta", key=np.abs, ascending=False)

print("Intercept (beta0):", intercepto)
print()
print(df_formula.head(30))


terms = [f"{b:.6f} * {f}" for f, b in zip(var_finales, coefs)]
equation = "logit(p) = " + f"{intercepto:+.6f} + " + " + ".join(terms)
print()
print("Formula final:")
print(equation)
print()

#Curva ROC
fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_estimator(best_pipe, x_test, y_test, ax=ax)

# Añadir la línea base (un modelo que adivina al azar)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

ax.set_title("Curva ROC - Regresion Logistica")
ax.legend()
plt.grid(alpha=0.3)
plt.show()


# Nombre del archivo pkl
archivo_modelo = "mejor_modelo_exam.pkl"

# Crear archivo pkl
with open(archivo_modelo, "wb") as archivo:
    pickle.dump(best_pipe, archivo)