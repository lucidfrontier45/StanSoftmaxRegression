import pickle
from hashlib import md5
from pathlib import Path

import numpy as np
from pystan import StanModel
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin


def append_const(x, v=0):
    N = len(x)
    return np.column_stack((np.zeros(N)+v, x))


def StanModel_cache(model_code, model_name="anon_model", model_dir="~/.stan"):
    model_dir = Path(model_dir).expanduser()
    if not model_dir.exists():
        model_dir.mkdir()

    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)

    cache_file: Path = model_dir.joinpath(cache_fn)
    if cache_file.exists():
        print("use cached stan model")
        with cache_file.open("rb") as fp:
            sm = pickle.load(fp)
    else:
        print("compile stan model")
        sm = StanModel(model_code=model_code, model_name=model_name)
        with cache_file.open(mode="wb") as fp:
            pickle.dump(sm, fp, pickle.HIGHEST_PROTOCOL)
    return sm


dir_path = Path(__file__).parent
model_file = dir_path.joinpath("model.stan")
with model_file.open("rt") as fp:
    model_code = fp.read()
    stan_model = StanModel_cache(model_code, "softmax_regression")


class StanSoftmaxRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, stan_model: StanModel = stan_model,
                 mode="sampling", s=10, **kwds):
        self.stan_model = stan_model
        self.mode = mode
        self.s = s
        self.stan_opts = kwds

    def fit(self, X, y):
        X = append_const(X, 1)
        y = np.asarray(y) + 1
        N, D = X.shape
        K = max(y)
        self.K_ = K
        data = {"N": N, "D": D, "K": K, "X": X, "y": y, "s": self.s}
        if self.mode == "sampling":
            fit = self.stan_model.sampling(data=data, **self.stan_opts)
            self.w_ = fit["w_raw"]
        elif self.mode == "vb":
            fit = self.stan_model.vb(data=data, **self.stan_opts)
            d = dict(zip(fit["sampler_param_names"], fit["sampler_params"]))
            self.w_ = np.column_stack([
                d["w_raw[{},{}]".format(i, j)]
                for i in range(1, K)
                for j in range(1, D+1)]
            ).reshape(-1, K-1, D)

    def predict_proba(self, X):
        X = append_const(X, 1)
        n_post = len(self.w_)
        z_raw = self.w_.dot(X.T)
        probs = []
        for i in range(n_post):
            z = append_const(z_raw[i].T, 0)
            proba = softmax(z, 1)
            probs.append(proba)

        return np.mean(probs, 0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(1)
