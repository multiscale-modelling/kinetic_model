import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

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


class AdsorptionKNNSlopeExtrap(AdsorptionKNN):
    """
    - Inherits all of AdsorptionKNN (log–KNN for type 0; raw–KNN for type 1).
    - For n<=10: exactly the base-class behavior.
    - For each distinct (pressure,temperature):
        * type 0 & n>10: extrapolate by the 2-point slope in log-space between n=8 and n=10.
        * type 1 & n>10: hold flat at the prediction for n=10.
    """
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # 1) get the vanilla KNN preds for everything
        base_preds = super().predict(df)
        df2 = df.reset_index(drop=True)
        preds = base_preds.copy()

        # 2) group by (pressure,temperature,adsorption_type_num)
        grp_inds = df2.groupby(
            ['pressure_bar','temperature','adsorption_type_num']
        ).groups

        for (p, T, ads_t), indices in grp_inds.items():
            idxs = list(indices)

            # TYPE 0 extrapolation
            if ads_t == 0:
                # find the single preds at n=8 and n=10
                mask8  = (df2['n']==8)
                mask10 = (df2['n']==10)
                y8s  = [base_preds[i] for i in idxs if mask8.loc[i]]
                y10s = [base_preds[i] for i in idxs if mask10.loc[i]]
                if len(y8s)==1 and len(y10s)==1 and not np.isnan(y8s[0]) and not np.isnan(y10s[0]):
                    ℓ8  = np.log(y8s[0])
                    ℓ10 = np.log(y10s[0])
                    slope = (ℓ10 - ℓ8) / 2.0  # over Δn = 2
                    # apply to every idx with n>10 in this group
                    for i in idxs:
                        n_i = df2.at[i,'n']
                        if n_i > 10:
                            ℓn = ℓ10 + slope*(n_i - 10)
                            preds[i] = np.exp(ℓn)

            # TYPE 1 flat continuation
            else:
                # get the single pred at n=10
                mask10 = (df2['n']==10)
                y10s = [base_preds[i] for i in idxs if mask10.loc[i]]
                if len(y10s)==1 and not np.isnan(y10s[0]):
                    y10 = y10s[0]
                    for i in idxs:
                        if df2.at[i,'n'] > 10:
                            preds[i] = y10

        return preds

    __call__ = predict
