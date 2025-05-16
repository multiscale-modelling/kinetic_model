import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import joblib

class AdsorptionKNN(BaseEstimator):
    """
    A grouped KNN regressor for adsorption data that scales inputs and
    maintains one KNN per combination of `n` and `adsorption_type_num`.
    Supports serialization via joblib.dump and loading via joblib.load.
    """
    def __init__(self, n_neighbors=2, weights='distance', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.scaler = StandardScaler()
        # keys will be tuples of (n, adsorption_type_num)
        self.knn_models = {}

    def fit(self, data: pd.DataFrame, y_col: str = 'mol_uc_avg'):
        """
        Fit the scaler on ['pressure_bar','temperature'] and build a KNN per group.
        data must include columns ['pressure_bar','temperature','n','adsorption_type_num', y_col].
        """
        # Fit global scaler
        features = data[['pressure_bar', 'temperature']].values
        self.scaler.fit(features)

        # Group by both n and adsorption_type_num
        groups = data.groupby(['n', 'adsorption_type_num'])
        for (n_val, ads_type), group in groups:
            X_grp = group[['pressure_bar', 'temperature']].values
            y_grp = group[y_col].values
            X_scaled = self.scaler.transform(X_grp)

            knn = KNeighborsRegressor(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                metric=self.metric
            )
            knn.fit(X_scaled, y_grp)
            self.knn_models[(n_val, ads_type)] = knn
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict adsorption values for df with columns ['pressure_bar','temperature','n','adsorption_type_num'].
        Returns array of predictions or np.nan for unseen (n,adsorption_type_num) pairs.
        """
        preds = []
        for _, row in df.iterrows():
            key = (row['n'], row['adsorption_type_num'])
            model = self.knn_models.get(key)
            if model is None:
                preds.append(np.nan)
            else:
                pt = np.array([[row['pressure_bar'], row['temperature']]])
                pt_scaled = self.scaler.transform(pt)
                preds.append(model.predict(pt_scaled)[0])
        return np.array(preds)

    # Allow loaded model to be called directly
    __call__ = predict


