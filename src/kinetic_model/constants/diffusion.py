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
          2) Fit scaler on temperature
          3) For each n, fit LinearRegression(y_norm ~ scaled T)
        """
        df = data.copy()
        # 1) normalize
        df['y_norm'] = df[D_col] / np.sqrt(df['temperature'])

        # 2) fit scaler on temperature
        T_vals = df[['temperature']].to_numpy()
        self.scaler.fit(T_vals)

        # 3) fit per-group models
        for n_val in df['n'].unique():
            mask = df['n'] == n_val
            X = df.loc[mask, ['temperature']].to_numpy()
            Xs = self.scaler.transform(X)
            y = df.loc[mask, 'y_norm'].to_numpy()

            lr = LinearRegression()
            lr.fit(Xs, y)
            self.models[n_val] = lr

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict raw D_TST for df with columns ['temperature','n'].
        Unseen n ⇒ np.nan.
        """
        preds = []
        for _, row in df.iterrows():
            model = self.models.get(row['n'])
            if model is None:
                preds.append(np.nan)
            else:
                X = np.array([[row['temperature']]])
                Xs = self.scaler.transform(X)
                y_norm = model.predict(Xs)[0]
                # back-transform
                preds.append(y_norm * np.sqrt(row['temperature']))
        return np.array(preds)

    # allow call like diff_model(df)
    __call__ = predict

class DiffusionLinearExtrapPlaceHolder(DiffusionLinear):
    """
    Subclass of DiffusionLinear that:
      - Uses the base-class linear-per-n models for n <= 10.
      - For any n > 10, it predicts D_TST at n = 6, 8, and 10,
        then scales each by sqrt(M_x / M_n) where M_n is the mass
        for that base model and M_x is the mass for the query n,
        and finally returns the average of those scaled values.
    """
    @staticmethod
    def _mass_calc(n: int) -> float:
        """
        Calculate the mass of a molecule based on the number of atoms.
        """
        return 1.008 * 2 * n + 12.01 * n

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # first get the base predictions (will be nan for n>10)
        base_preds = super().predict(df)
        preds = base_preds.copy()

        for i, row in df.iterrows():
            n = int(row['n'])
            T = row['temperature']

            # if n <= 10, we already have the correct base_preds[i]
            if n <= 10:
                continue

            # n > 10: extrapolate using models at 6, 8, 10
            Mx = self._mass_calc(n)
            scaled_vals = []
            for base_n in (6, 8, 10):
                model = self.models.get(base_n)
                if model is None:
                    continue
                # predict normalized diffusion and back-transform
                Xs = self.scaler.transform([[T]])
                y_norm = model.predict(Xs)[0]
                D_base = y_norm * np.sqrt(T)
                Mn = self._mass_calc(base_n)
                scaled_vals.append(D_base * np.sqrt(Mx / Mn))

            if scaled_vals:
                preds[i] = float(np.mean(scaled_vals))
            else:
                preds[i] = np.nan

        return preds

    __call__ = predict