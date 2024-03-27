import xarray as xr, numpy as np, xarray_helper as xrh, pandas as pd
from typing import Tuple
from nptyping import NDArray as A, Shape as S

# def monitor_time(f):
#     def new_f(*args, **kwargs):
#         import time
#         start = time.time()
#         r = f(*args, **kwargs)
#         end= time.time()
#         print(f"f duration: {end-start}")
#         return r
#     f.timed = new_f
#     return f

# @monitor_time
# def test(a, b):
#     return a+b

# r = test.timed(0, 1)
# print(r)

import scipy.signal

d= xr.Dataset()
d["t"] = xr.DataArray(np.arange(0, 100), dims=["t"])
d["t2"] = xr.DataArray(np.arange(0, 29), dims=["t"], coords=dict(t=np.arange(1, 30)))
d["x"] = xr.DataArray(np.linspace(0, 100, 51), dims=["x"])
d["y"] = xr.DataArray(np.linspace(0, 100, 51), dims=["y"])

d["test"] = d["t"] * d["x"] +d["y"]
print(d)
# exit()

@xrh.xrfunc(in_dims=dict(a=["values"]), vectorized=True)
def mean(a):
    # print(np.shape(a))
    return np.mean(a, axis=-1)

@xrh.xrfunc(in_dims=dict(a=["values"]), out_dims=[["values"]], vectorized=True)
def id(a):
    # print(np.shape(a))
    return a

class NanError(Exception): pass

@xrh.xrfunc(in_dims=dict(X=["samples", "feats"], y=["samples"]))
def score(X, y):
    # if lambda X, y: ((~np.isnan(X)).all(axis=1)  | (~np.isnan(y))).any():
    #     raise NanError()
    import sklearn.linear_model
    return sklearn.linear_model.LinearRegression().fit(X, y).score(X, y)

@xrh.xrfunc(in_dims=dict(x=["t"]), out_dims=(["f"], ["t_bin"], ["f", "t_bin"]))
def spectrogram(x, **kwargs):
    import scipy.signal
    f, t, pxx = scipy.signal.spectrogram(x, **kwargs)
    return f, t, pxx

@xrh.xrfunc(in_dims=dict(args=["t"], kwargs=["t2"]))
def mean_sum(*args, **kwargs):
    return np.sum([np.mean(a) for a in args] + [np.mean(a) for a in kwargs.values()])



# spectrogram = xrh.xrfunc(in_dims=dict(x=["t"]), out_dims=tuple(["f"], ["t_bin"], ["f", "t_bin"]))(scipy.signal.spectrogram)

# @xrh.xrfunc
# def spectrogram(x: Arr["t"], *args) -> Tuple[Arr["f"], Arr["t_bin"], Arr["f", "t_bin"]]:

#     import scipy.signal
#     f, t, pxx = scipy.signal.spectrogram(x, *args)
#     return f, t, pxx


# d["test"].xrh.spectrogram(t=["t"], fs=d["fs"])

d["mean"] = mean.xr.map_dims(values=["x", "y"]).exec(d["test"])
d["id"] = id.xr.map_dims(values="x").exec(d["test"])
d[["f", "t_bin", "spectrogram"]] = spectrogram.xr.exec(d["test"], fs=3)
d["score"] = score.xr.map_dims(samples=["x"], feats="y").exec(d["test"], d["x"])
d["mean_sum"] = mean_sum.xr.map_dims(t=["x"], t2=["y"]).exec(d["x"], d["x"], y=d["y"], z=d["y"])



# d["mean"] = d["test"].mean(dim="t")
# d["r2"] = test2.dims(feats=["x", "t"], samples=["y"]).run(d["test"], d["y"])


print(d)