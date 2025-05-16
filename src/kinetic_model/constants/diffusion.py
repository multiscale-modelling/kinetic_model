import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

class DiffusionLinear(BaseEstimator):
    """
    Grouped linear regressor for diffusion data:
      - Internally normalizes D_TST by sqrt(T),
      - Scales T with a global StandardScaler,
      - Fits one LinearRegression per n,
      - Predict returns raw D_TST.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.models: dict[int, LinearRegression] = {}

    def fit(self, data: pd.DataFrame, D_col: str = 'D_TST') -> 'DiffusionLinear':
        """
        Fit one linear model per n:
          1) Compute y_norm = D_TST / sqrt(T)
          2) Fit scaler on T
          3) For each n, fit LinearRegression(y_norm ~ scaled T)
        """
        df = data.copy()
        # 1) normalize
        df['y_norm'] = df[D_col] / np.sqrt(df['T'])

        # 2) fit scaler on T
        T_vals = df[['T']].to_numpy()
        self.scaler.fit(T_vals)

        # 3) fit per-group models
        for n_val in df['n'].unique():
            mask = df['n'] == n_val
            X = df.loc[mask, ['T']].to_numpy()
            Xs = self.scaler.transform(X)
            y = df.loc[mask, 'y_norm'].to_numpy()

            lr = LinearRegression()
            lr.fit(Xs, y)
            self.models[n_val] = lr

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict raw D_TST for df with columns ['T','n'].
        Unseen n ⇒ np.nan.
        """
        preds = []
        for _, row in df.iterrows():
            model = self.models.get(row['n'])
            if model is None:
                preds.append(np.nan)
            else:
                X = np.array([[row['T']]])
                Xs = self.scaler.transform(X)
                y_norm = model.predict(Xs)[0]
                # back-transform
                preds.append(y_norm * np.sqrt(row['T']))
        return np.array(preds)

    # allow call like diff_model(df)
    __call__ = predict