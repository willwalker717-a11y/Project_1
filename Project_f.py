from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def grid(train, test):
    
        
     def Train(df, df2):
            
         df, df2 = pd.read_csv(df),pd.read_csv(df2)
        
         def all_descriptors(smiles):
              mol = Chem.MolFromSmiles(smiles)
              if mol is None:
                  return {}
            
              return Descriptors.CalcMolDescriptors(mol)
           
         def featurize(F, Y):
               selector = SelectKBest(f_regression, k=10)
               selector.fit_transform(F, Y)
               selected_features = F.columns[selector.get_support()]
               
               return selected_features
           
         def prep(df):
        
               Imputer = SimpleImputer(strategy="median")
               Imputer.set_output(transform="pandas")
               Features = pd.DataFrame(df["SMILES"].apply(all_descriptors).tolist(), index=df.index)
               Features = Features.drop(columns="MolLogP")
               Features = Features.replace([np.inf, -np.inf], np.nan)
               Features = Imputer.fit_transform(Features)
               Y = df["label"]
               return Features, Y
    
         F, Y = prep(df)
         F2, Y2 = prep(df2)
           
         Fea = featurize(F,Y)
           
         F = F[Fea]
         F2 = F2[Fea]
           
         return F, Y, F2, Y2
        
        
        
    
     pipe = Pipeline([
            ("scaler", StandardScaler()),   # placeholder; will be swapped by grid search
            ("pca", PCA()),
            ("svr", SVR())
            ])
        
     param_grid = {
            "scaler": [
                StandardScaler(),
                MinMaxScaler(),
                RobustScaler(),
                Normalizer()
                ],
            "pca__n_components": [1,5,10],
            "svr__kernel": ["linear", "rbf"],
            }
        
     grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            refit=True
            )
        
     X_train,Y_train,X_test,Y_test = Train(train,test)
        
     grid.fit(X_train,Y_train)
     best_model = grid.best_estimator_
        
     Y_pred = best_model.predict(X_test)
     return Y_pred, Y_test
    

Predicted, Real = grid("freesolv_train.csv", "freesolv_test.csv")

print("optimized MAE: " + str(mean_absolute_error(Real, Predicted)))

plt.figure()
plt.scatter(Real, Predicted)

# Add labels
plt.xlabel("Experimental Value")
plt.ylabel("Calculated Value")

# Show plot
plt.show()
