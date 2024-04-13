import re
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from numpy import trapz
# from pprint import pprint
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn import metrics
# from sklearn.cluster import MiniBatchKMeans
from imblearn import FunctionSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
# from imblearn.under_sampling import TomekLinks, AllKNN, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
# from imblearn.under_sampling import CondensedNearestNeighbour, NeighbourhoodCleaningRule, OneSidedSelection
# from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.pipeline import make_pipeline
# from imblearn.under_sampling import ClusterCentroids
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay, roc_auc_score, balanced_accuracy_score
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from pdpbox import pdp
import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

df = pd.read_csv("/home/larissa.ayumi/german_raw.csv",delimiter=',')

df.isna().sum().sum()

cost_matrix = {0:1, 1:5}
threshold = 0.5
alpha = 1

# Encoding

df_encoder = df.copy()
# -1 mal pagador
df_encoder.rename(columns={'GoodCustomer':'TARGET','Gender':'CODE_GENDER'},inplace=True)

df_encoder.TARGET = list(map(lambda x: 1 if x == -1 else  0 ,df_encoder.TARGET))

df_encoder.TARGET.value_counts()

df_encoder.CODE_GENDER.value_counts()[1]/(df_encoder.CODE_GENDER.value_counts()[0] + 
                                          df_encoder.CODE_GENDER.value_counts()[1])

cat_cols = df_encoder.loc[:,(df_encoder.dtypes == object).values].columns # selecionadas apenas Gender e PurposeOfLoan
num_cols = df_encoder.loc[:,(df_encoder.dtypes == 'int64')|(df_encoder.dtypes == float)].columns # selecionadas as variáveis numéricas não categóricas 

from sklearn.preprocessing import LabelEncoder
# Male == 1/ Female == 0
label_encoder = LabelEncoder()
for col in cat_cols:
    df_encoder[col] = label_encoder.fit_transform(df_encoder[col])

target_column_name = 'TARGET'
sensitive_column_name = 'CODE_GENDER'
pos_label = 0

# Split Data
X, y = df_encoder.drop(columns=[target_column_name]),df_encoder[target_column_name]
X_train, x_aux, y_train, y_aux = train_test_split(X, y, random_state=42,test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_aux,y_aux,test_size=0.50, random_state=42)

# XGBOOST
study_dict = {'learning_rate': 0.08730015238141663, 'max_depth': 8, 'subsample': 0.9885334531348575, 
              'colsample_bytree': 0.8860893884596323, 'gamma': 0.38560298254941716, 
              'reg_alpha': 0.44299387597591205, 'reg_lambda': 0.37110498550700155, 
              'min_child_weight': 0.9339053330213946}

xg = xgb.XGBClassifier(learning_rate = study_dict['learning_rate'],max_depth = study_dict['max_depth'],
                               subsample = study_dict['subsample'],colsample_bytree = study_dict['colsample_bytree'],
                               gamma = study_dict['gamma'],
                               reg_lambda = study_dict['reg_lambda'],min_child_weight = study_dict['min_child_weight'],
                               random_state= 42)
xg.fit(X_train,y_train)

# SHAP
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(xg)
shap_values = explainer(X_train)

# PDP
def create_message(feature_importance):
    mes = "Em um modelo de machine learning criado para predizer se a pessoa é um mau ou bom cliente explique como cada variável influência na variável target para o caso específico desse cliente, sendo que quanto maior o valor absoluto SHAP dela mais ela influência  na variável target e se este valor for negativo,  quer dizer que essa feature influência negativamente na variável target levando ela a se aproximar do zero, o que significa que a pessoa ser considerada um bom cliente, caso o valor SHAP for positivo quer dizer que essa feature está aproximando a variável target do 1 que é o cliente ser considerado um mau cliente. Sabendo disso,  considere esse exemplo: " + feature_importance + " Avalie como cada uma dessas features está influenciando na target, baseado na explicação fornecida"
    return mes

def feature_importance(index, number_of_features):
    mes = ""
    # Sort features based on importance scores
    sorted_features = X_train.columns[np.argsort(abs(shap_values.values[index]))[::-1]]
    sorted_shap_values = np.argsort(abs(shap_values.values[599]))[::-1]
    for i in range (sorted_features[:number_of_features].shape[0]):
        mes += f"{sorted_features[i]}: {shap_values.values[index][sorted_shap_values[i]]}, "
    return mes

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

message = create_message(feature_importance(2,5))
sequences = pipeline(
message,
max_length=10000,
do_sample=True,
top_k=10,
num_return_sequences=1,
truncation=True,
eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(seq["generated_text"])
