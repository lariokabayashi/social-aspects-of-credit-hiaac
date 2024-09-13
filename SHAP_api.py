import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import optuna
import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.preprocessing import LabelEncoder

from flask import Flask, request, jsonify
import shap

app = Flask(__name__)

def get_shap_results(client_index):
    df = pd.read_csv("/home/larissa.ayumi/german_raw.csv",delimiter=',')   
    df_encoder = df.copy()
    # -1 mal pagador
    df_encoder.rename(columns={'GoodCustomer':'TARGET','Gender':'CODE_GENDER'},inplace=True)
    target_column_name = 'TARGET'
    sensitive_column_name = 'CODE_GENDER'
    pos_label = 0
    df_encoder.TARGET = list(map(lambda x: 1 if x == -1 else  0 ,df_encoder.TARGET))
    cat_cols = df_encoder.loc[:,(df_encoder.dtypes == object).values].columns # selecionadas apenas Gender e PurposeOfLoan
    from sklearn.preprocessing import LabelEncoder
    # Male == 1/ Female == 0
    # 'Electronics' = 2, 'Education' = 1, 'Furniture' = 3, 'NewCar' = 5, 'UsedCar' = 9, 'Business' = 0, 'HomeAppliances' = 4, 'Repairs' = 7, 'Other' = 6, 'Retraining'= 8
    label_encoder = LabelEncoder()
    for col in cat_cols:
        df_encoder[col] = label_encoder.fit_transform(df_encoder[col])
    X, y = df_encoder.drop(columns=[target_column_name]),df_encoder[target_column_name]
    X_train, x_aux, y_train, y_aux = train_test_split(X, y, random_state=42,test_size=0.4)

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
    mes = ""
    sorted_feature_indices = np.argsort(abs(shap_values.values[int(client_index)]))[::-1]
    sorted_features = X_train.columns[sorted_feature_indices]
    for i in range (0,sorted_feature_indices.shape[0]):
        mes += f"{sorted_features[i]}: {shap_values.values[int(client_index)][sorted_feature_indices[i]]}, "
    return mes

@app.route('/shap', methods=['GET'])
def results():
    client_index = request.args.get('client_index', '')
    if not client_index:
        return jsonify({'error': 'Missing client index'}), 400
    results = get_shap_results(client_index)
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=4444)