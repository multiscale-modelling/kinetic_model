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


class EnsembleModel:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def predict(self, X):
        y1 = self.model1(X)
        y2 = self.model2(X)
        return (y1 + y2) / 2

    __call__ = predict


import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

class AdsorptionGP(BaseEstimator):
    """
    A grouped Gaussian Process regressor for adsorption data that scales inputs and
    maintains one GP per combination of `n` and `adsorption_type_num`.
    Uses a Matérn ν=0.5 kernel (exponential kernel) with ARD, plus signal and noise kernels.
    """
    def __init__(self,
                 kernel=None,
                 alpha=0.0,
                 normalize_y=True,
                 length_scale=[1.0, 1.0],
                 length_scale_bounds=(1e-8, 1e5),
                 signal_level=1.0,
                 signal_bounds=(1e-3, 1e3),
                 noise_level=1e-5,
                 noise_bounds=(1e-8, 1e1),
                 n_restarts_optimizer=5):
        # ARD Matern kernel + signal scaling + white noise
        if kernel is None:
            matern = Matern(length_scale=length_scale,
                            length_scale_bounds=[length_scale_bounds]*2,
                            nu=0.5)
            signal = C(signal_level, signal_bounds)
            noise = WhiteKernel(noise_level=noise_level,
                                noise_level_bounds=noise_bounds)
            self.kernel = signal * matern + noise
        else:
            self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.scaler = StandardScaler()
        # store one GP model per (n, adsorption_type_num)
        self.gp_models = {}

    def fit(self, data: pd.DataFrame, y_col: str = 'mol_uc_avg'):
        features = data[['pressure_bar', 'temperature']].values
        self.scaler.fit(features)

        groups = data.groupby(['n', 'adsorption_type_num'])
        for (n_val, ads_type), group in groups:
            X_grp = group[['pressure_bar', 'temperature']].values
            y_grp = group[y_col].values
            X_scaled = self.scaler.transform(X_grp)

            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts_optimizer
            )
            gp.fit(X_scaled, y_grp)
            self.gp_models[(n_val, ads_type)] = gp
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            key = (row['n'], row['adsorption_type_num'])
            gp = self.gp_models.get(key)
            if gp is None:
                preds.append(np.nan)
            else:
                pt = np.array([[row['pressure_bar'], row['temperature']]])
                pt_scaled = self.scaler.transform(pt)
                pred, std = gp.predict(pt_scaled, return_std=True)
                preds.append(pred[0])
        return np.array(preds)

    __call__ = predict


class AdsorptionGPSlopeExtrap(AdsorptionGP):
    """
    Same as AdsorptionGP for n<=10.
    For type 0 & n>10: extrapolate via log-slope from n=8 to n=10.
    For type 1 & n>10: hold flat at n=10.
    """
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        base_preds = super().predict(df)
        df2 = df.reset_index(drop=True)
        preds = base_preds.copy()

        groups = df2.groupby(['pressure_bar', 'temperature', 'adsorption_type_num']).groups
        for (p, T, ads_t), idxs in groups.items():
            idxs = list(idxs)
            if ads_t == 0:
                mask8 = df2['n'] == 8
                mask10 = df2['n'] == 10
                y8 = [base_preds[i] for i in idxs if mask8.loc[i]]
                y10 = [base_preds[i] for i in idxs if mask10.loc[i]]
                if len(y8)==1 and len(y10)==1 and not np.isnan(y8[0]) and not np.isnan(y10[0]):
                    l8, l10 = np.log(y8[0]), np.log(y10[0])
                    slope = (l10 - l8) / 2.0
                    for i in idxs:
                        n_i = df2.at[i, 'n']
                        if n_i > 10:
                            preds[i] = np.exp(l10 + slope * (n_i - 10))
            else:
                mask10 = df2['n'] == 10
                y10 = [base_preds[i] for i in idxs if mask10.loc[i]]
                if len(y10)==1 and not np.isnan(y10[0]):
                    val10 = y10[0]
                    for i in idxs:
                        if df2.at[i, 'n'] > 10:
                            preds[i] = val10

        return preds

    __call__ = predict





## ======================= Poly1DModel for adsorption capacity ======================

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import least_squares

R = 8.314462618  # J/mol/K

# ============================= Peng–Robinson (ethylene) =============================

@dataclass
class PREoS:
    Tc: float = 282.33  # K
    Pc: float = 5.02e6  # Pa
    omega: float = 0.089

    def _a_b(self):
        a = 0.45724 * (R*self.Tc)**2 / self.Pc
        b = 0.07780 * R*self.Tc / self.Pc
        return a, b

    def _z_phi_scalar(self, P, T):
        a, b = self._a_b()
        kappa = 0.37464 + 1.54226*self.omega - 0.26992*self.omega**2
        alpha = (1 + kappa*(1 - np.sqrt(T/self.Tc)))**2
        A = a*alpha*P/(R**2*T**2)
        B = b*P/(R*T)
        coeff = [1, -(1-B), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)]
        roots = np.roots(coeff)
        Z = np.max(roots.real[np.isclose(roots.imag, 0)])
        lnphi = (Z - 1) - np.log(Z - B) - (A/(2*np.sqrt(2)*B)) * \
                np.log((Z + (1+np.sqrt(2))*B)/(Z + (1-np.sqrt(2))*B))
        return Z, np.exp(lnphi)

    def fugacity(self, P, T):  # Pa
        P = np.atleast_1d(P).astype(float)
        T = np.atleast_1d(T).astype(float)
        f = np.empty_like(P, dtype=float)
        for i, (p, t) in enumerate(zip(P, T)):
            _, phi = self._z_phi_scalar(p, t)
            f[i] = phi * p
        return f

    def rho_bulk(self, P, T):  # mol/m^3
        # optional, for "out" if you want it later
        P = np.atleast_1d(P).astype(float)
        T = np.atleast_1d(T).astype(float)
        Z = np.empty_like(P, dtype=float)
        for i, (p, t) in enumerate(zip(P, T)):
            z, _ = self._z_phi_scalar(p, t)
            Z[i] = z
        return P/(Z*R*T)

# ================================ Langmuir helpers =================================

class LangmuirEq:
    @staticmethod
    def n_abs(f, nmax, b):
        return nmax * (b*f) / (1.0 + b*f)

    @staticmethod
    def henry_from_params(nmax, b):
        return nmax * b  # KH

    @staticmethod
    def f_half(b):
        return 1.0 / max(b, 1e-30)

# =============================== Per-T isotherm fit ================================

class PerTFitter:
    """
    Fit single-site Langmuir independently at every temperature:
    n = nmax * b f / (1 + b f), where f = fugacity(P,T).
    """
    def __init__(self, eos: PREoS, nmax_lb: float = 1.0, b_bounds=(1e-12, 1e-3)):
        self.eos = eos
        self.nmax_lb = nmax_lb
        self.b_bounds = b_bounds

    def _fit_one_T(self, P, T, n):
        f = self.eos.fugacity(P, np.full_like(P, float(T)))
        # inits
        nmax0 = max(1.05*np.max(n), self.nmax_lb + 1e-9)
        k = min(3, len(f))
        idx = np.argsort(f)[:k]
        if k >= 2:
            slope = np.polyfit(f[idx], n[idx], 1)[0]
            slope = max(slope, 1e-18)
        else:
            slope = max(n[0]/max(f[0], 1e-18), 1e-18)
        b0 = np.clip(slope / max(nmax0, 1e-18), self.b_bounds[0], self.b_bounds[1])

        def resid(x):
            return LangmuirEq.n_abs(f, x[0], x[1]) - n

        lb = np.array([self.nmax_lb, self.b_bounds[0]], float)
        ub = np.array([np.inf,        self.b_bounds[1]], float)
        x0 = np.minimum(np.maximum(np.array([nmax0, b0], float), lb+1e-12), ub-1e-12)

        res = least_squares(resid, x0, bounds=(lb, ub), loss='linear')
        nmax, b = float(res.x[0]), float(res.x[1])
        n_fit = LangmuirEq.n_abs(f, nmax, b)
        rmse = float(np.sqrt(np.mean((n_fit - n)**2)))
        KH = LangmuirEq.henry_from_params(nmax, b)
        fhalf = LangmuirEq.f_half(b)
        return nmax, b, KH, fhalf, rmse, f, n_fit

    def fit(self, df: pd.DataFrame):
        assert {'pressure','temperature','absolute'}.issubset(df.columns)
        rows, preds = [], []
        for T, g in df.sort_values(['temperature','pressure']).groupby('temperature'):
            P = g['pressure'].to_numpy(float)
            n = g['absolute'].to_numpy(float)
            nmax, b, KH, fhalf, rmse, f, n_fit = self._fit_one_T(P, float(T), n)
            rows.append(dict(temperature=float(T), nmax=nmax, b=b, KH=KH, f_half_Pa=fhalf, rmse=rmse))
            preds.append(pd.DataFrame({'pressure':P, 'temperature':float(T), 'pred_absolute':n_fit}))
        params_df = pd.DataFrame(rows).sort_values('temperature').reset_index(drop=True)
        df_out = df.merge(pd.concat(preds, ignore_index=True), on=['pressure','temperature'], how='left')
        return params_df, df_out

# ========================== Polynomial T→parameter models ==========================

class Poly1DModel:
    """
    y(T) ≈ poly_d( (T - mu)/sig ), optionally in log-space (for strictly positive y).
    """
    def __init__(self, degree=5, log_space=False):
        self.degree = degree
        self.log_space = log_space
        self.mu = None
        self.sig = None
        self.coef = None  # highest power first (np.polyfit style)

    def _tx(self, T):
        T = np.asarray(T, float)
        return (T - self.mu) / self.sig

    def fit(self, T, y):
        T = np.asarray(T, float)
        y = np.asarray(y, float)
        self.mu = float(np.mean(T))
        self.sig = float(np.std(T) if np.std(T) > 0 else 1.0)
        X = self._tx(T)
        if self.log_space:
            y = np.log(np.clip(y, 1e-300, None))
        self.coef = np.polyfit(X, y, self.degree)
        return self

    def predict(self, T):
        X = self._tx(T)
        val = np.polyval(self.coef, X)
        if self.log_space:
            return np.exp(val)
        return val

    def as_dict(self, name):
        return dict(name=name, degree=self.degree, log_space=self.log_space, mu=self.mu, sig=self.sig, coef=self.coef.tolist())

# ======================= Global P–T isotherm from poly params ======================

class PTIsothermPolyModel:
    """
    - Fit polynomials nmax(T), b(T) from per-T Langmuir fits.
    - Predict absolute at any (P,T) using fugacity from PREoS.
    - Predict out = rho_bulk(P,T) * Vp (Vp learned from data, identical to your snippet).
    - Predict excess = absolute - out.
    """
    def __init__(self, eos: PREoS, degree_nmax=5, degree_b=5, log_b=False, Vp=None):
        self.eos = eos
        self.nmax_model = Poly1DModel(degree=degree_nmax, log_space=False)
        self.b_model    = Poly1DModel(degree=degree_b,    log_space=log_b)
        self.Vp = Vp  # m^3/g (or whatever matches your units)

    # --- training from per-T results ---
    def fit_from_params(self, params_df: pd.DataFrame):
        T = params_df['temperature'].to_numpy(float)
        self.nmax_model.fit(T, params_df['nmax'].to_numpy(float))
        self.b_model.fit(T, params_df['b'].to_numpy(float))
        return self

    # --- identical OUT logic to your previous code ---
    def set_Vp_from_df(self, df: pd.DataFrame):
        P = df['pressure'].to_numpy(float)
        T = df['temperature'].to_numpy(float)
        if 'capacity_out' in df.columns:
            y = df['capacity_out'].to_numpy(float)
        else:
            y = (df['absolute'].to_numpy(float) - df['excess'].to_numpy(float))
        rho = self.eos.rho_bulk(P, T)                         # mol/m^3
        self.Vp = float(np.dot(rho, y) / (np.dot(rho, rho) + 1e-18))  # m^3/g
        if self.Vp < 0: self.Vp = 0.0
        return self.Vp

    # --- predictions ---
    def _nmax_b_T(self, T):
        T = np.atleast_1d(T).astype(float)
        return self.nmax_model.predict(T), self.b_model.predict(T)

    def predict_absolute(self, P, T):
        P = np.atleast_1d(P).astype(float); T = np.atleast_1d(T).astype(float)
        f = self.eos.fugacity(P, T)
        nmax, b = self._nmax_b_T(T)
        return nmax * (b*f) / (1.0 + b*f)

    def predict_out(self, P, T):
        assert self.Vp is not None, "Call set_Vp_from_df(...) first or pass Vp in ctor."
        return self.eos.rho_bulk(P, T) * self.Vp

    def predict_excess(self, P, T):
        return self.predict_absolute(P, T) - self.predict_out(P, T)

    # convenience
    def add_predictions(self, df, prefix="pred_"):
        P = df['pressure'].to_numpy(float)
        T = df['temperature'].to_numpy(float)
        out = df.copy()
        out[f'{prefix}absolute'] = self.predict_absolute(P, T)
        if self.Vp is not None:
            out[f'{prefix}out'] = self.predict_out(P, T)
            out[f'{prefix}excess'] = out[f'{prefix}absolute'] - out[f'{prefix}out']
        return out

