import pandas as pd
df = pd.read_csv('./data/bodyPerformance_no_class.csv')
print( df.shape )
print( df.head() )

cat_cols = [ 'gender' ]
for col in cat_cols :
    df[ col ] = df[ col ].astype( 'category' )
X = df.drop( columns=[ 'body fat_%' ] )
Y = df[ 'body fat_%' ]

print( X.shape )   # (13393, 10)
print( Y.shape )   # (13393,)

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split( X , Y , test_size=0.2 , random_state=42 )

import optuna 
from lightgbm import LGBMRegressor, early_stopping 
import numpy as np

optuna.logging.set_verbosity( optuna.logging.WARNING )

def objective( trial ) : 
    from sklearn.model_selection import KFold
    cv = KFold( n_splits=5 , shuffle=True , random_state=42 )
    cv_r2_scores = [ ]
    best_interations = [ ]
    for train_idx , val_idx in cv.split( X_train , Y_train ) :
        X_tr , X_val = X_train.iloc[ train_idx ] , X_train.iloc[ val_idx ]
        Y_tr , Y_val = Y_train.iloc[ train_idx ] , Y_train.iloc[ val_idx ]
        model = LGBMRegressor( 
            n_estimators=1000 , 
            learning_rate=trial.suggest_float( 'learning_rate' , 0.01 , 0.1 )  ,
            num_leaves=trial.suggest_int( 'num_leaves' , 10, 30 ),
            max_depth=trial.suggest_int( 'max_depth' , 3 , 10 ),
            min_child_samples=trial.suggest_int( 'min_child_samples', 10 , 50 ),
            subsample=trial.suggest_float( 'subsample' , 0.6 , 1.0 ) , 
            subsample_freq=1, verbose=-1,random_state=42, n_jobs=-1 , 
            colsample_bytree=trial.suggest_float('colsample_bytree',0.6,1.0),
        )
        model.fit( X_tr , Y_tr , eval_set=[ ( X_val , Y_val ) ] ,
                    callbacks=[ early_stopping( stopping_rounds=30 , verbose=False ) ] )
        r2 = model.score( X_val , Y_val )
        cv_r2_scores.append( r2 )
        best_interations.append( model.best_iteration_ )
    trial.set_user_attr( 'best_n_estimators' , int( np.mean( best_interations ) ) )
    return np.mean( cv_r2_scores )

sampler = optuna.samplers.TPESampler( seed=42 )
study = optuna.create_study( direction='maximize' , sampler=sampler )
study.optimize( objective , n_trials=20 )
print( '\n[ 1. Optuna 튜닝 결과 ]' )
print( '최적 하이퍼파라미터:' , study.best_params )
optional_n_estimators = study.best_trial.user_attrs[ 'best_n_estimators' ]
print( '최적 트리:' , optional_n_estimators )

final_model = LGBMRegressor(
    random_state=42 , verbose=-1 , n_jobs=-1 ,
    **study.best_params , 
    n_estimators=optional_n_estimators
)
final_model.fit( X_train , Y_train )
print( '\n[ 2. 모델 성능 ]' )
print( 'Train R2:' , final_model.score( X_train , Y_train ) )
print( 'Test R2:' , final_model.score( X_test , Y_test ) )
print( '\n[ 3. 피처 중요도 ]' )
importance = pd.Series( final_model.feature_importances_ , index=final_model.feature_name_ )
print( importance )

Y_pred = final_model.predict( X_test )
result = pd.DataFrame( { '실제' : Y_test.values , '예측' : Y_pred } )
result[ '차이' ] = result[ '예측' ] - result[ '실제' ]
print( '\n[ 4. 실제값 vs 예측값 (test 앞 10명) ]' )
print( result.head( 10 ) )
print( '평균 오차:' , np.mean( np.abs( result[ '차이' ] ) ) )

import joblib
joblib.dump( final_model , './body_model.pkl' )

test_r2 = final_model.score( X_test , Y_test )
joblib.dump( test_r2 , './body_r2.pkl' )

model = joblib.load( './body_model.pkl' )

new_bodyPerformance = pd.DataFrame( [
    # 1번 고객
    { 'age': 27 , 'gender': 'M' , 'height_cm': 172.3 , 'weight_kg': 75.24 ,
      'diastolic': 80 , 'systolic': 130 , 'gripForce': 54.9 ,
      'sit and bend forward_cm': 18.4 , 'sit-ups counts': 60 , 'broad jump_cm': 217 } ,
    # 2번 고객
    { 'age': 45 , 'gender': 'F' , 'height_cm': 158.0 , 'weight_kg': 62.0 ,
      'diastolic': 78 , 'systolic': 125 , 'gripForce': 24.0 ,
      'sit and bend forward_cm': 15.0 , 'sit-ups counts': 25 , 'broad jump_cm': 140 }
] )
for col in cat_cols :
    new_bodyPerformance[ col ] = new_bodyPerformance[ col ].astype( 'category' )
    preds = model.predict( new_bodyPerformance )
    print( '\n[ 5. 신규 데이터 예측 ]' )
print( '(1번고객)예측 체지방률(%):' , preds[0] )
print( '(2번고객)예측 체지방률(%):' , preds[1] )



