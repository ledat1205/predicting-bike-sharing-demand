import catboost as cb
import pandas as pd
import numpy as np

def load_data(train_dir):
    df_train = pd.read_csv(train_dir)
    return df_train

def processing(df):
    if type(df) != pd.DataFrame :
        df = pd.DataFrame(df, columns=['id', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt'])

    #remove id
    df = df.iloc[:,1:]

    #switch yr
    df[["yr"]] = np.where(df[['yr']]==0, -1, df[['yr']])
    
    #one-hot weather
    weather = {'good': [0], 'mist': [0], 'rain': [0], 'pour': [0]}
    #for 1 mark when inference
    if df.shape[0] == 1:
        temp = int(df['weathersit'].values)    
        weather[list(weather)[temp-1]] = [1]
        w = pd.DataFrame.from_dict(weather)
        df = df.join(w)

    else:
        #for dataframe train
        dummy_weathersit = pd.get_dummies(df['weathersit'])
        dummy_weathersit = dummy_weathersit.rename(columns = {1: 'good', 2: 'mist', 3: 'rain', 4: 'pour'})
        df = pd.concat([df, dummy_weathersit], axis=1)
        
    


    #features construction
    #add rush hour feature
    rush_hours_conditions = [
                            df['workingday'].eq(1) & df['hr'].ge(7) & df['hr'].le(9),
                            df['workingday'].eq(1) & df['hr'].ge(16) & df['hr'].le(20),
                        
    ]
    choices = [1,1]
    df['peak'] = np.select(rush_hours_conditions, choices, default=0)

    return df

df_train = load_data(train_dir='trainDataset.csv')
df_train = processing(df_train)


train_features = df_train[df_train.columns.difference(['cnt'])]
train_labels = df_train['cnt']

train_dataset = cb.Pool(train_features, np.log1p(train_labels)) 
catboost_model = cb.CatBoostRegressor(iterations=300,border_count=100, depth=10,l2_leaf_reg=0.7,learning_rate=0.1,loss_function='RMSE',
                             colsample_bylevel=0.85, 
                             verbose=False)

catboost_model.fit(train_dataset)

def predict(mark):
    mark = np.insert(mark,13,0)
    mark = np.expand_dims(mark,axis=0)
    mark = processing(mark)
    mark = mark[mark.columns.difference(['cnt'])]

    y_predict = catboost_model.predict(mark)
    y_predict = np.exp(y_predict) - 1
    y_predict = y_predict.round()

    return y_predict
