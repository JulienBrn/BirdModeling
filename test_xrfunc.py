import xarray as xr, numpy as np, xarray_helper as xrh, pandas as pd
from typing import Tuple
from nptyping import NDArray as A, Shape as S


from time import sleep
from tqdm.auto import tqdm
import logging, beautifullogger

logger = logging.getLogger(__name__)
beautifullogger.setup()




# with tqdm(range(3)) as outer:
#     inner_total = 300
#     with tqdm(total=inner_total, leave=False) as inner:
#         for i in outer:
#             inner.reset(inner_total)  # reinitialise without clearing
#             for j in range(inner_total):
#                 sleep(0.01)
#                 inner.update()
#             inner.refresh()
#             logger.info("test")
# print("yes")
# exit()    


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
d["t"] = xr.DataArray(np.arange(0, 1000000), dims=["t"])
# d["t2"] = xr.DataArray(np.arange(0, 29), dims=["t"], coords=dict(t=np.arange(1, 30)))
d["x"] = xr.DataArray(np.linspace(0, 100, 50), dims=["x"])
d["y"] = xr.DataArray(np.linspace(0, 100, 50), dims=["y"])


d["test"] = d["t"] * d["x"] +d["y"]
# d = d.chunk(chunks=dict(t=100000, x=3, y=3))
print(d)
# print(d["test"].chunks)
# print(d)
# tmp = xr.align(d["y"], d["x"])
# print(tmp)
# exit()
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
    # print("shape", x.shape)
    # print("type", type(x))
    # x = x.compute()
    import scipy.signal
    f, t, pxx = scipy.signal.spectrogram(x, **kwargs)
    # print(pxx.shape)
    return f, t, pxx

@xrh.xrfunc(in_dims=dict(args=["t"], kwargs=["t2"]))
def mean_sum(*args, **kwargs):
    return np.sum([np.mean(a) for a in args] + [np.mean(a) for a in kwargs.values()])


from scipy.signal.windows import gaussian

@xrh.xrfunc(in_dims=dict(sig=["t"]), out_dims=(["f"], ))
def stft_f(stft):
    return stft.f

@xrh.xrfunc(in_dims=dict(), out_dims=(["t_bin"], ))
def stft_t(stft, n_sig):
    return stft.f

@xrh.xrfunc(in_dims=dict(t=["t"], win=["win"]), out_dims=([], ["f"], ["t_stft"]))
def stft(t, win, *stft_args, t_precision=None, **stft_kwargs):
    if t_precision is None:
        t_precision = t.mean()/10**(-8)
    period =t[1] - t[0]
    if not (np.abs(t[1:] - t[: -1] - period) <= t_precision).all():
        raise Exception(f"Not regular fs")
    fs = 1/period
    stft = scipy.signal.ShortTimeFFT(win, *stft_args, fs=fs, **stft_kwargs)
    return stft, stft.f, stft.t(t.size) + t[0]

@xrh.xrfunc(in_dims=dict(sig=["t"]), out_dims=(["f", "t_stft"],), output_dtypes=[float], vectorized=True)
def stft_spectrogram(stft, sig):
    if len(stft.shape) == 0:
        stft = stft.item()
        # print(sig.shape)
    return stft.spectrogram(sig)
        

# spectrogram = xrh.xrfunc(in_dims=dict(x=["t"]), out_dims=tuple(["f"], ["t_bin"], ["f", "t_bin"]))(scipy.signal.spectrogram)

# @xrh.xrfunc
# def spectrogram(x: Arr["t"], *args) -> Tuple[Arr["f"], Arr["t_bin"], Arr["f", "t_bin"]]:

#     import scipy.signal
#     f, t, pxx = scipy.signal.spectrogram(x, *args)
#     return f, t, pxx


# d["test"].xrh.spectrogram(t=["t"], fs=d["fs"])
logger.info("Starting")
d["mean"] = mean.xr.map_dims(values=["x", "y"]).exec(d["test"])
# d["id"] = id.xr.map_dims(values="x").exec(d["test"])
d[["stft", "f", "t_stft"]] = stft.xr(d["t"], win = gaussian(50, std=8, sym=True), hop=1000)
print(d)
d["spectrogram"] = stft_spectrogram.xr.set_model(d)(d["stft"], d["test"])

# d[["f", "t_bin", "spectrogram"]] = spectrogram.xr.exec(d["test"], fs=3)
# d["score"] = score.xr.map_dims(samples=["x"], feats="y").exec(d["test"], d["x"])
# d["mean_sum"] = mean_sum.xr.map_dims(t=["x"], t2=["y"]).exec(d["x"], d["x"], y=d["y"], z=d["y"])



# d["mean"] = d["test"].mean(dim="t")
# d["r2"] = test2.dims(feats=["x", "t"], samples=["y"]).run(d["test"], d["y"])
print(d)
from dask.diagnostics import ProgressBar
ProgressBar().register()
d = d.compute()
logger.info("Done")
print(d)
input()

def tmp(f, mappings, *args, **kwargs):
    new_args = []
    new_f = f
    new_mapping = []
    #such that new_f(*new_args) is equivalent to f(*args, **kwargs)
    # and new_mapping[k] = mapping[new_args[k]]
