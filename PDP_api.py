import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from sklearn.inspection import partial_dependence
import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.preprocessing import LabelEncoder

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

def get_PDP_results(feature_name):
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

    results = partial_dependence(xg, X, [feature_name])
    return tuple(zip(results["grid_values"][0].tolist(), results["average"].tolist()[0]))

@app.route('/results', methods=['GET'])
def results():
    feature_name = request.args.get('feature', '')
    if not feature_name:
        return jsonify({'error': 'Missing feature name'}), 400
    results = get_PDP_results(feature_name)
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=4000)
