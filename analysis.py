
# ============================================================================ #
# Imports


# Standard imports
from pathlib import Path
import json

# Third-party imports
import numpy as np
import pandas as pd
from elephant import kernels
from elephant.statistics import instantaneous_rate
import quantities as pq
from neo import SpikeTrain
from scipy import signal
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ============================================================================ #
# Loading Data

all_folders = list(Path("/media/julien/data1/CarmenData/").iterdir())
target_folder =all_folders[1]
unit_name = 'ch12#2'

inputs_folder = target_folder / 'Inputs'
extracted_files_folder = target_folder / 'Extracted_files'


f = open(target_folder / "Inputs/metadata.json")
metadata = json.load(f)
print(f'Available units are: {list(metadata["units_analysis"].keys())}')

if not unit_name in metadata["units_analysis"].keys():
    raise Exception(f"Invalid unit {unit_name} of folder {target_folder}")
print(f"Using unit {unit_name} of folder {target_folder}")
# Sampling Frequency per stream
fs_ap = metadata["fs_ap"]  # in Hz
fs_mic = metadata["fs_mic"]  # in Hz
fs_bua = metadata["real_ds_fs"]  # in Hz

# Recording start and end info
rec_start = metadata["rec_start"]
rec_end = metadata["rec_end"]

# Labels ifo
sybs_labels = metadata["sybs_labels"]

# Song channel
song_channel = metadata["song_channel"]



# Song labels
labels_df = pd.read_csv(
    (inputs_folder / 'labels.txt'), header=None, names=["beg", "end", "syb"]
)

# Song array
song = np.load(extracted_files_folder/ (song_channel + '.npy'))

# ============================================================================ #
# Unit windows


# Define name


## Unit-related info
unit_window = metadata["units_windows"][0][unit_name]
unit_window_beg = unit_window[0]["beginning"]
unit_window_end = unit_window[0]["end"]
bp = metadata["units_bp"][unit_name]
if not isinstance(bp, list):
    bp = [bp]

original_unit_spk_times_in_sec = np.loadtxt(extracted_files_folder / (unit_name + '.txt'))

def correct_spike_train(spike_times, unit_window_beg, unit_window_end, rec_end):
    """Corrects unit and labels according to unit window.
        Carmen looked if the unit seemed stable or mostly lost in the cumulative plot; if not, 
        a window for analysis was added to the metadata and the spikes outside this window should be deleted
    """
    if unit_window_end == -1:
        unit_window_end = rec_end
        
    # Get unit spiketimes and waveforms
    corrected_unit_times = np.where(
        np.logical_and(spike_times >= unit_window_beg, spike_times <= unit_window_end) == True
    )
    new_spike_times = spike_times[corrected_unit_times]
    
    return new_spike_times

unit_spk_times_in_sec = correct_spike_train(spike_times=original_unit_spk_times_in_sec, 
                                            unit_window_beg=unit_window_beg, 
                                            unit_window_end=unit_window_end, 
                                            rec_end=rec_end )

# bua_ch_name = 'CSC18'

# bua_trace = np.load(extracted_files_folder / (bua_ch_name + '_envelope_1000Hz.npy'))

# ============================================================================ #
# IFR computation


kernel_sigma = 10*pq.ms # decided by Arthur
sampling_period_ifr = 1*pq.ms

kernel = kernels.GaussianKernel(sigma=kernel_sigma)

st = SpikeTrain(
    unit_spk_times_in_sec,
    t_stop=rec_end,
    units="s",
    sampling_rate=fs_ap * pq.Hz,
)  # necessary to convert to use elephant

ifr = instantaneous_rate(st, sampling_period=sampling_period_ifr, kernel=kernel) # downsampled to 1000Hz
ifr_trace = ifr.magnitude.ravel()
ifr_fs = ifr.sampling_rate.rescale(('Hz'))


# Here is the detrending bit. It's not obvious that detendring is the best option because we might bias the spiketrain. 
# Therefore, Arthur would like an option to do or not do it easily if recomputing the final results
# So far, Carmen would look at the cumulative of the spikes and see if there would be a part that seemed to need detrending, declaring the bp range in the metadata
# However, to keep the part that we don't want to detrend intact, we concatenate the detrended part to the original part and make sure they match in the y-axis by adding the mean

# ============================================================================ #
# Selecting subpart

bp_in_sp = (bp * ifr_fs.magnitude).astype(int)


if type(bp_in_sp) is not list:
    det_sig = signal.detrend(ifr_trace, bp=np.array(bp_in_sp))
    
     # shifting it above 0 to avoid negative values in analysis
    try:
        if len(bp) ==1:
            det_part_to_keep = det_sig[bp_in_sp[0]:] + np.mean(ifr_trace)
            ifr_trace[bp_in_sp[0]:] = det_part_to_keep
        else:
            det_part_to_keep = det_sig[bp_in_sp[0]:bp_in_sp[1]] + np.mean(ifr_trace) 
            ifr_trace[bp_in_sp[0]:bp_in_sp[1]] = det_part_to_keep
    except:
        print(f"bp is {bp}")
        raise

# ============================================================================ #
# Converting to xarray


pitch_lower_bound = metadata["pitch_lower_bounds"]
pitch_upper_bound = metadata["pitch_upper_bounds"]

import xarray as xr, xarray_helper as xrh
dataset = labels_df[["syb", "beg", "end"]].to_xarray().rename(index="syllable_index")
dataset["song_data"] = xr.DataArray(song[:, 0], dims="song_index", coords=dict(song_index=range(song.size)))
dataset["song_fs"] = fs_mic
dataset["song_t"] = dataset["song_index"]/dataset["song_fs"]

print(dataset)
print(ifr_trace.shape)

dataset["ifr"] = xr.DataArray(ifr_trace, dims="ifr_index", coords=dict(ifr_index=range(ifr_trace.size)))
dataset["ifr_fs"] = ifr_fs
dataset["ifr_t"] = dataset["ifr_index"]/dataset["ifr_fs"]
dataset = dataset.set_coords(["ifr_t", "song_t", "ifr_fs", "song_fs"])
print(dataset)


# dataset["ifr"].plot()
# plt.show()
# ============================================================================ #
# Filtering according to motif



dataset = dataset.assign_coords(motif=xr.DataArray(["azd", "abcd", "zzzzz"], dims="motif"))
def compute_motif(syllables: np.ndarray, motif: str): 
    n = len(motif)
    motif = xr.DataArray(list(motif), dims="motif_index")
    shifted_syllables = xr.DataArray(np.stack([np.roll(syllables, -shift) for shift in range(n)]), dims=["motif_index", "syllable_index"])
    w = (shifted_syllables==motif).all(dim="motif_index")
    ret = w.where(w).ffill("syllable_index", n-1)
    ret = ret.cumsum().where(~ret.isnull())-(w.cumsum()-1)*n -1
    return ret.to_numpy()
        
dataset["motif_index"] = xr.apply_ufunc(compute_motif, 
    dataset["syb"], dataset["motif"], input_core_dims=[["syllable_index"], []], 
    output_core_dims=[["syllable_index"]], vectorize=True
)

dataset = dataset.set_coords(["motif_index"])

cond = ~dataset["motif_index"].isnull().all("motif")
for k in dataset.data_vars.keys():
    if set(cond.dims).issubset(set(dataset[k].dims)):
        dataset[k] = dataset[k].where(cond)
dataset = dataset.dropna(dim="syllable_index", how="all")
print(dataset)

# ============================================================================ #
# Splitting the syllable

def split(beg, end, size, offset):
    new_beg = beg+offset
    ids = np.array(range(int((end-new_beg)/size)))
    print(len(ids))
    ids = np.pad(ids, (0, 300-len(ids)), mode="constant", constant_values=-1)
    return ids

max_subsyllable = ((dataset["end"] - dataset["beg"] + (0.015 * dataset["song_fs"]))/(0.1 * dataset["song_fs"])).max()
dataset["subsyllable"] = xr.DataArray(np.arange(max_subsyllable), dims=["subsyllable"])
dataset["subsyllable_beg"] = dataset["beg"] + (0.015 * dataset["song_fs"]) + (0.1 * dataset["song_fs"]) * dataset["subsyllable"]
dataset["subsyllable_end"] = dataset["beg"] + (0.015 * dataset["song_fs"]) + (0.1 * dataset["song_fs"]) * (dataset["subsyllable"]+1)
print(dataset)


# ============================================================================ #
# Grouping by same syllable

def change_grp(x):
    x = x.rename(syllable_index = "syb_num").drop(["syb"])
    x["syb_num"] = x["syb_num"].notnull().cumsum()
    return x

xr.Dataset.nicegroupby = xrh.nicegroupby
dataset = dataset.nicegroupby("syb").map(change_grp)
print(dataset)

# ============================================================================ #
# Computing song features

progress = tqdm(desc="pitch feature")
progress.total = dataset["subsyllable_beg"].size

def compute_pitch(a):
    global progress
    progress.update(1)
    from pitch import processPitch
    return processPitch(a, dataset["song_fs"].item(), 1000, 10000)
def compute_amp(a): 
    return np.mean(a)
def compute_entropy(a): 
    from scipy.signal import welch
    f, p = welch(a)
    p /= np.sum(p)
    power_per_band_mat = p[p > 0]
    spectral_mat = -np.sum(power_per_band_mat * np.log2(power_per_band_mat))
    return spectral_mat

feats = []
for f in tqdm([compute_pitch, compute_amp, compute_entropy], desc="Computing song feats"):
    feat: xr.DataArray = xr.apply_ufunc(lambda s, b, e: f(s[int(b):int(e)]) if not np.isnan([b, e]).any() else np.nan, 
        dataset["song_data"], dataset["subsyllable_beg"], dataset["subsyllable_end"], 
        input_core_dims=[["song_index"]] + [[]]*2, vectorize=True)
    feat = feat.expand_dims(song_feat=[f.__name__.replace("compute_", "")])
    feats.append(feat)


dataset["song_feats"] = xr.concat(feats, dim="song_feat")
print(dataset)

# ============================================================================ #
# Rescaling features

def scale_standard(a):
    import sklearn.preprocessing
    scaler = sklearn.preprocessing.StandardScaler()
    ashape = a.shape
    a = np.moveaxis(a, -1, 0)
    a = np.reshape(a, (a.shape[0], -1))
    scaler = scaler.fit(a)
    ret = scaler.transform(a)
    ret = np.reshape(ret, (ret.shape[0], ) + ashape[:-1])
    ret = np.moveaxis(ret, 0, -1)
    return ret

def scale_none(a):
    return a

feats = []
for scaling in [scale_standard]:
    feat: xr.DataArray = xr.apply_ufunc(scaling, dataset["song_feats"], input_core_dims=[["syb_num"]], output_core_dims=[["syb_num"]])
    feat = feat.expand_dims(scale_method=[scaling.__name__.replace("scale_", "")])
    feats.append(feat)

dataset["song_feats_scaled"] = xr.concat(feats, dim="scale_method")
print(dataset)

# ============================================================================ #
# Dimension reduction

def reduce_pca(a):
    if np.isnan(a).all():
        return np.full_like(a, np.nan)
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA()
    mask = np.isnan(a)
    a = np.where(mask, np.nanmean(a, axis=0), a)
    return np.where(mask, np.nan, pca.fit_transform(a))

def reduce_none(a):
    return a

feats = []
for reduce in [reduce_pca]:
    feat: xr.DataArray = xr.apply_ufunc(reduce, dataset["song_feats_scaled"], input_core_dims=[["syb_num", "song_feat"]], output_core_dims=[["syb_num", "song_feat"]], vectorize=True)
    feat = feat.expand_dims(reduce_method=[reduce.__name__.replace("reduce_", "")])
    feats.append(feat)

dataset["feats_reduced"] = xr.concat(feats, dim="reduce_method")
print(dataset)

# ============================================================================ #
# IFR per syllable

def get_syb_ifr(ifr, start, end):
    if not np.isnan([start, end]).any():
        return np.mean(ifr[int(start):int(end)])
    return np.nan

dataset["lag_s"] = xr.DataArray(np.linspace(0.020, 0.100, 9), dims="lag_s")

dataset["syb_ifr"] = xr.apply_ufunc(get_syb_ifr,
    dataset["ifr"], 
    dataset["ifr_fs"] * (dataset["subsyllable_beg"]/dataset["song_fs"] - dataset["lag_s"]), 
    dataset["ifr_fs"] * (dataset["subsyllable_end"]/dataset["song_fs"] - dataset["lag_s"]), 
    input_core_dims=[["ifr_index"]] + [[]]*2, vectorize=True)

print(dataset)

# ============================================================================ #
# Fitting reduction
import sklearn.linear_model, sklearn.mixture
def dict_to_array(dim, d):
    a = xr.DataArray(np.array(list(d.values())), dims=[dim], coords={dim: np.array(list(d.keys()))})
    return a

def get_score(feats, ifr, m):
    mask = np.isnan(ifr)
    ifr = ifr[~mask]
    feats = feats[~mask, :]
    return m.fit(feats, ifr).score(feats, ifr)


dataset["model"] = dict_to_array("model_name", dict(
    # glm= sklearn.linear_model.PoissonRegressor(), 
    linear=sklearn.linear_model.LinearRegression(), 
    # bgmm=sklearn.mixture.BayesianGaussianMixture(),
    # gmm=sklearn.mixture.GaussianMixture(),
    # bgmm3=sklearn.mixture.BayesianGaussianMixture(n_components=3),
    # gmm3=sklearn.mixture.GaussianMixture(n_components=3)
))
                                               
dataset["score"] = xr.apply_ufunc(get_score, dataset["feats_reduced"], dataset["syb_ifr"], dataset["model"], input_core_dims=[["syb_num", "song_feat"]] + [["syb_num"]]+[[]], vectorize=True)
print(dataset)
# dataset["score"].plot(col="syb")





# ============================================================================ #
# Computing bootstrap

dataset["bt_index"] = xr.DataArray(np.arange(100), dims="bt_index")

progress=tqdm()
progress.total = dataset["score"].size*dataset["bt_index"].size

def bootstrap_score(feats, ifr, m, i):
    mask = np.isnan(ifr)
    ifr = ifr[~mask]
    feats = feats[~mask, :]
    global progress
    shuffled_ifr = np.random.default_rng().permuted(ifr)
    res= m.fit(feats, shuffled_ifr).score(feats, shuffled_ifr)
    progress.update()
    return res

dataset["bt_score"] = xr.apply_ufunc(bootstrap_score, dataset["feats_reduced"], dataset["syb_ifr"], dataset["model"], dataset["bt_index"], input_core_dims=[["syb_num", "song_feat"]] + [["syb_num"]]+[[]]*2, vectorize=True)
dataset["ratio"] = (dataset["score"] - dataset["bt_score"])/dataset["score"]
print(dataset)
print(dataset["ratio"])



# ============================================================================ #
# Computing kde
def kde(a, p):
    import scipy.stats
    return scipy.stats.gaussian_kde(a)(p)

dataset["kde_points"] = xr.DataArray(np.linspace(-1, 2, 1000), dims="kde_points")
dataset["bt_score_kde"] = xr.apply_ufunc(kde, dataset["bt_score"], dataset["kde_points"], input_core_dims=[["bt_index"]] + [["kde_points"]], output_core_dims=[["kde_points"]], vectorize=True)

# ============================================================================ #
# Plotting


f, axs = plt.subplots(dataset["syb"].size, dataset["subsyllable"].size, sharex=True, sharey=True)

first = True
for i in range(dataset["syb"].size):
    for j in range(dataset["subsyllable"].size):
        for k in range(dataset["lag_s"].size):
            kw = dict(label=f'lag={int(dataset["lag_s"].isel(lag_s =k).item()*1000)}ms') if first else {}
            (
                dataset["bt_score_kde"].isel(syb=i, subsyllable=j).isel(lag_s=k, drop=True)
                .sel(scale_method="standard", reduce_method="pca", model_name="linear", drop=True)
                .drop(["song_fs", "ifr_fs"]).plot.line(ax=axs[i, j], x="kde_points", color=f"C{k}", **kw)
            )
            axs[i, j].axvline(x=dataset["score"].isel(syb=i, subsyllable=j, lag_s=k).item(), color=f"C{k}")
        first =False

f.legend()
f.tight_layout()

# import xarray.plot as xrp

# fg = xrp.FacetGrid(dataset, row="syb", col="subsyllable")
# f = dataset["bt_score_kde"].plot(row="syb", col="subsyllable")
# fg.map(lambda *args, **kwargs: print("Args", args, kwargs), x=dataset["kde_points"], y=dataset["bt_score_kde"])
# print(dataset["score"])
plt.show()
# dataset = xr.apply_ufunc(split, 
#     dataset["beg"], dataset["end"], 
#     (0.1 * dataset["song_fs"]).astype(int), (0.015 * dataset["song_fs"]).astype(int), 
#     output_core_dims=[["subsyllable"]], vectorize=True)

# print(dataset)

# import matplotlib.pyplot as plt

# dataset["ifr"].plot(x="ifr_t")
# # dataset["song_data"].plot(x="song_t")
# plt.show()