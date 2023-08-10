# Then, create your new samples of silica nanoparticles in Main_Independent_Dataset_COLAB.csv with Publication_id = 150
# PLEASE UPLOAD the new file (Main_Independent_Dataset_COLAB.csv) containing your new samples of silica nanoparticles by running this code

import pandas as pd
import catboost
import io
from catboost import CatBoostClassifier
print('The CatBoost version is {}.'.format(catboost.__version__))

# Predict your new samples of silica nanoparticles by running this code, and PLEASE WAIT until the results have been downloaded 

your_new_samples_id = 150
independent_id = 116
data = pd.read_csv('Main_Indepedent_Dataset_COLAB (4).csv', encoding='unicode_escape').iloc[:,:15]

def convert(Cell_viability):
    if Cell_viability == 'Cytotoxic':
        return 1
    if Cell_viability == 'Non_cytotoxic':
        return 0    
    else:
        return ''
    
def convert_back(Cell_viability):
    if Cell_viability == 1:
        return 'Cytotoxic'
    if Cell_viability == 0:
        return 'Non_cytotoxic'
    else:
      return ''
    
data['convert'] = data['Cell_viability'].apply(convert)
data = data.drop('Cell_viability', axis=1)
data = data.rename(columns={'convert':'Cell_viability'}) 

shuffled_main_dataset = data[data['Publication_id'] < independent_id].sample(frac=1, random_state=2022)
independent_dataset = data[data['Publication_id'] >= independent_id]
data = pd.concat([shuffled_main_dataset, independent_dataset])

X = pd.get_dummies(data.drop('Cell_viability', axis=1))
X = X.drop([
    'SiO$_{2}$NP_medium_serum_15%_FBS',
    'Cell_morphology_microglia',

    'Cell_organ_heart',

    'Cell_id_MPMC/3t3',

    'Surface_modification_CHO',
    'Hydrodynamic_size_water_nm_not_determined',
    'Cell_source_hamster',
    'Assay_viability_Sytox_Red',

    'Viability_indicator_live_cell',
], axis=1)

y = data[['Cell_viability', 'Publication_id']]
y = y[y['Publication_id'] < independent_id]
y = y.drop('Publication_id',axis=1)
y = y.to_numpy().ravel()

y_test = data[['Cell_viability','Publication_id']]
y_test = y_test[y_test['Publication_id'] >= independent_id]
y_test =  y_test.drop('Publication_id',axis=1)
y_test = y_test.values

X_test = X[X['Publication_id'] >= independent_id]
X_test = X_test.drop('Publication_id',axis=1)
X_test = X_test.sort_index(ascending=True)
X_test_shap = X_test
X_test = X_test.values.reshape(-1,len(X_test.columns))

X = X[X['Publication_id'] < independent_id]
X = X.drop('Publication_id',axis=1)

models = {"CatBoost Classifier": CatBoostClassifier(learning_rate= 0.05,max_depth= 7,random_state=2022)}

for name, model in models.items():
    model.fit(X.values,y)
    preds = model.predict(X_test)
    
    df_preds = pd.DataFrame(preds)
    df_preds['predicted'] = df_preds[0]
    ytestt = pd.DataFrame(y_test)
    ytestt['observed'] = ytestt[0]
    result = pd.concat([ytestt, df_preds], axis=1)
    result = result.drop([0,0], axis=1)
    independent_data = data[data['Publication_id'] >= independent_id].sort_index(ascending=True).reset_index()
    independent_data = pd.DataFrame(independent_data)
    result = pd.concat([result, independent_data], axis=1).drop(['Cell_viability'], axis=1)
    result['observed'] = result['observed'].apply(convert_back)
    result['predicted'] = result['predicted'].apply(convert_back)
    result[result['Publication_id'] >= your_new_samples_id].to_csv('CatBoost_your_result.csv')
    files.download('CatBoost_your_result.csv')
    result.to_csv('CatBoost_result.csv')
    files.download('CatBoost_result.csv')
