import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error


def prepare_data(X: pd.DataFrame,
                 y: pd.Series,
                 num_features: list,
                 scale_X: bool = False, 
                 scale_y: bool = False,
                 scaler_type: str = "standard",   # "standard" ou "minmax"
                 test_size: float = 0.2,
                 random_state: int = 42):
    
    #=========================================================
    # Fonction de préparation de données avec un scale des features numérique et split train/test
    #=========================================================
 
    # Toutes les autres colonnes déjà encodées
    OneHot_features = [col for col in X.columns if col not in num_features]
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    
    # initialisation du scale
    Scaler = MinMaxScaler if scaler_type.lower() == "minmax" else StandardScaler
    scaler_X = Scaler() if scale_X else None
    scaler_y = Scaler() if scale_y else None

    # scale X
    if scale_X:
        #X_train_num = pd.DataFrame(scaler_X.fit_transform(X_train[num_features]))
        #X_test_num = pd.DataFrame(scaler_X.transform(X_test[num_features]))
        X_train_num =scaler_X.fit_transform(X_train[num_features])
        X_train_num = pd.DataFrame(X_train_num, columns=num_features,index=X_train.index)
        
        X_test_num = scaler_X.transform(X_test[num_features])
        X_test_num = pd.DataFrame(X_test_num,columns=num_features,index=X_test.index)

    else:
        X_train_num = X_train[num_features].copy()
        X_test_num = X_test[num_features].copy()

    # Reconstruction de X final
    X_train_final = pd.concat([X_train_num, X_train[OneHot_features]], axis=1)
    X_test_final = pd.concat([X_test_num, X_test[OneHot_features]], axis=1)
    

    # scale y
    if scale_y:
        y_train_final = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_final = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
    else:
        y_train_final = y_train.values
        y_test_final = y_test.values

    return X_train_final, X_test_final, y_train_final, y_test_final, scaler_X, scaler_y


#======================================================================================
#======================================================================================
#======================================================================================

def train_evaluate_model(model,
                         X_train,
                         X_test,
                         y_train,
                         y_test,
                         scaler_y=None,
                         plot=True,
                         title = None):
    #=========================================================
    # Fonction d'entrainement et d'évaluation du modèle 
    #=========================================================
    
    # fit
    model.fit(X_train,y_train)
    
    #=================================
    # predict
    y_hat_train = model.predict(X_train) 
    y_hat_test = model.predict(X_test)    
    
    #=================================
    # y real inverse of y scale
    if scaler_y is not None:
        if scaler_y is np.log1p:
            y_train_real = np.expm1(np.array(y_train))
            y_test_real = np.expm1(np.array(y_test))
            y_hat_train_real = np.expm1(np.array(y_hat_train))
            y_hat_test_real = np.expm1(np.array(y_hat_test)) 
        else:
            y_train_real = scaler_y.inverse_transform(np.array(y_train).reshape(-1, 1)).ravel()
            y_test_real  = scaler_y.inverse_transform(np.array(y_test).reshape(-1, 1)).ravel()
            y_hat_train_real = scaler_y.inverse_transform(np.array(y_hat_train).reshape(-1, 1)).ravel()
            y_hat_test_real  = scaler_y.inverse_transform(np.array(y_hat_test).reshape(-1, 1)).ravel()
    else:
        y_train_real = np.array(y_train)
        y_test_real  = np.array(y_test)
        y_hat_train_real = np.array(y_hat_train)
        y_hat_test_real  = np.array(y_hat_test)
    
    #=================================
    # metrics
    
    # coefficient de détermination R2
    r2_train = r2_score(y_train_real, y_hat_train_real)
    r2_test  = r2_score(y_test_real,  y_hat_test_real)
    
    # Root Mean Squared Error & Mean Absolute Error
    rmse = root_mean_squared_error(y_test_real, y_hat_test_real)
    mae  = mean_absolute_error(y_test_real, y_hat_test_real)
    #=================================
    # results    
    results = {"model": model,
               "R2_train": r2_train,
               "R2_test": r2_test,
               "RMSE": rmse,
               "MAE": mae}
               #"y_hat": y_hat_test_real}
    #=================================
    # plot
    if plot:
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test_real, y_hat_test_real, alpha=0.3)

        #minv = min(y_test_real.min(), y_hat_test_real.min())
        #maxv = max(y_test_real.max(), y_hat_test_real.max())
        #plt.plot([minv, maxv], [minv, maxv], "r--")
        plt.plot([y_test_real.min(), y_test_real.max()],[y_test_real.min(), y_test_real.max()],'r--')
        plt.xlabel(r"Valeurs réelles $y_{test}$")
        plt.ylabel(r"Valeurs prédites $\hat{y}$")
        plt.title(title if title else f"y_test vs y_hat — {model.__class__.__name__}")
        plt.tight_layout()
        plt.grid(True)
        #plt.show()
    
    return results, y_hat_test_real
    
#======================================================================================
#======================================================================================
#======================================================================================

def grid_around_best(value,step):
    #=========================================================
    # Fonction qui crée une petite grille autour d'une valeur donnée par RandomSearchCV
    #=========================================================
    if value is None:
        return [None]

    #step = 10 if value > 100 else 5

    grid = np.arange(max(1, value - step), value + step + 1,1)
    
    if value == 'min_samples_split':
        grid = np.arange(max(2, value - step), value + step + 1,1)
    
    if value == 'subsample':
        grid = np.arange(max(0, value - step/10), min(1,value + step/10),0.05)
    

    return grid
    

#======================================================================================
#======================================================================================
#======================================================================================

def group_feature_name(feature):
    #=========================================================
    # Fonction qui regroupe les features afin d'éliminer le biais dans le calcul des features_importance_
    #=========================================================
    
#     if feature.startswith("Neighborhood_grouped"):
#         return "Neighborhood_grouped"
#     elif feature.startswith("PrimaryPropertyType_grouped"):
#         return "PrimaryPropertyType_grouped"
#     elif feature in ["mono_usage", "multi_usages"]:
#         return "usage_type"
#     elif feature in ["Electricity", "NaturalGas"]:
#         return "EnergySource"

    if feature.startswith('Neighborhood_grouped'):
        return 'Neighborhood_grouped'
    if feature.startswith('PropertyUseType'):
        return 'PropertyUseType'
    elif feature in ['mono_usage', 'multi_usages']:
        return 'usage_type'
    elif feature in ['ElectricityPercentage', 'NaturalGasPercentage','OtherEnergyPercentage']:
        return "EnergySource"
    else:
        return feature