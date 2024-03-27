import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib, concurrent.futures
from typing import List, Tuple, Dict, Any, Literal, Set, Sequence
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger, shutil

logger=logging.getLogger(__name__)
class DimRemoveExcpt(Exception):pass

def apply_file_func(func, in_folder, path: xr.DataArray, *args, out_folder=None, name = None, recompute=False, save_group=None, n=1, path_arg=None, n_ret=1, output_core_dims=None, n_outcore_nans=None,**kwargs):
    def subapply(*args):
        # print("entering subapply")
        nonlocal nb_nans, nb_already_computed
        paths: List[pathlib.Path]=list(args[:n])
        args=args[n:]
        if (not pd.isna(paths).any()) and not ("nan" in paths):
            if not out_folder is None:
                dest: pathlib.Path = pathlib.Path(out_folder)
                progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans, status="Checking already computed"))
                for path in paths:
                    dest=dest/pathlib.Path(path).relative_to(pathlib.Path(in_folder))
                dest=dest.with_suffix(".pkl")
                # print(out_folder, dest, paths)
                if dest.exists() and not recompute:
                    nb_already_computed+=1
                    progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans))
                    progress.total=progress.total-1
                    progress.update(0)
                    return str(dest)
            data=[]
            progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans, status="Loading input data"))
            for path in paths:
                in_path: pathlib.Path  = pathlib.Path(in_folder)/path
                match in_path.suffix:
                    case  ".pkl":
                        data.append(pickle.load(in_path.open("rb")))
                    case ".npy":
                        data.append(toolbox.np_loader.load(in_path))
                    case ext:
                        raise Exception(f"Unknown extension {ext} for {in_path} from {path} {pd.isna(path)} {type(path)}")
            progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans, status="Computing"))
            # print(dest, dest.exists())
            if path_arg is None:
                ret = func(*data, *args)
            else:
                ret = func(*data, *args, **{path_arg:paths})
            progress.update(1)
            try:
                if isinstance(ret, xr.DataArray):
                    is_na = False
                else:
                    is_na = pd.isna(ret)
            except:
                is_na = False
            try:
                if is_na:
                    nb_nans+=1
                    progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans))
            except:
                print(is_na)
                print(ret)
                raise
                # print("nan")
                # input()
            # print(f"Value considered {is_na}:\n{ret}")
            if not out_folder is None and not is_na:
                progress.set_postfix(dict(nb_nans=nb_nans, status="Dumping"))
                dest.parent.mkdir(exist_ok=True, parents=True)
                pickle.dump(ret, dest.with_suffix(".tmp").open("wb"))
                # print(out_folder, str(dest.with_suffix(".tmp")))
                # input()
                shutil.move(str(dest.with_suffix(".tmp")),str(dest))
                # print("Should exist !", dest, dest.exits())
                return str(dest)
            else:
                return ret
        else:
            # print("returning nan undefined input")
            
            if not output_core_dims is None:
                res = tuple(xr.DataArray(data=np.reshape([np.nan]*n_outcore_nans, [1]* (len(dims)-1)+[-1]), dims=dims) for dims in output_core_dims)
                # print(res)
                # input()

            else:
                res= tuple([np.nan for _ in range(n_ret)]) if not n_ret == 1 else np.nan
            return res
    if not save_group is None:
        group_path = pathlib.Path(save_group)
        if group_path.exists() and not recompute:
            return pickle.load(group_path.open("rb"))
    if float(path.count()) > 0:
        if name is None and not out_folder is None:
            progress = tqdm.tqdm(desc=f"Computing {out_folder}", total=float(path.count()))
        elif not name is None:
            progress = tqdm.tqdm(desc=f"Computing {name}", total=float(path.count()))
        else:
            progress = tqdm.tqdm(desc=f"Computing", total=float(path.count()))
    nb_nans=0
    nb_already_computed=0
    # print(args)
    # print(type(path))
    # print(path.ndim)
    progress.set_postfix(dict(status="Applying xarray ufunc"))
    res = xr.apply_ufunc(subapply, path, *args, vectorize=True, output_dtypes=None if out_folder is None else ([object]*n_ret), output_core_dims=output_core_dims if not output_core_dims is None else ((), ), **kwargs)
    if not save_group is None:
        group_path.parent.mkdir(exist_ok=True, parents=True)
        pickle.dump(res, group_path.with_suffix(".tmp").open("wb"))
        shutil.move(str(group_path.with_suffix(".tmp")),str(group_path))
    progress.update(0)
    return res

def nunique(a, axis, to_str=False):
            a = np.ma.masked_array(a, mask=pd.isna(a))
            if to_str:
                try:
                    a = a.astype(str)
                except Exception as e:
                    ta = a.reshape(-1)
                    for i in range(0, ta.size, 10):
                        try:
                            _ = ta[i:i+10].astype(str)
                        except:
                            e.add_note(f"Problem converting array values to string... Initial dtype is {a.dtype}. Example of values is\n{ta[i:i+10]} {ta[i+3]} {type(ta[i+3])}")
                            # print("Array is ", a)
                            raise e
                    e.add_note(f"Problem converting array values to string... Initial dtype is {a.dtype}. Example of values not found")
                    raise e
            sorted = np.ma.sort(a,axis=axis, endwith=True, fill_value=''.join([chr(255) for _ in range(5)]))
            unshifted = np.apply_along_axis(lambda x: x[:-1], axis, sorted)
            shifted = np.apply_along_axis(lambda x: x[1:], axis, sorted)
            diffs =  (unshifted != shifted)
            return np.ma.filled((diffs!=0).sum(axis=axis)+1, 1)


def auto_remove_dim(dataset:xr.Dataset, ignored_vars=None, kept_var=None, dim_list=None):
    def remove_numpy_dim(var: np.ndarray):
        nums = nunique(var, axis=-1, to_str=True)
        if (nums==1).all():
            return np.take_along_axis(var, np.argmax(~pd.isna(var), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
        else:
            raise DimRemoveExcpt("Can not remove dimension")
        
    ndataset = dataset
    if kept_var is None:
        vars = list(dataset.keys())+list(dataset.coords)
    else:
        vars = kept_var
    if not ignored_vars is None:
        vars = list(set(vars) - set(ignored_vars))
    # for var in set(list(dataset.keys())+list(dataset.coords)) - set(vars):
    #     ndataset[var] = dataset[var]
    for var in tqdm.tqdm(vars, desc="fit var dims", disable=True):
        # if var in ignored_vars:
        #     ndataset[var] = dataset[var]
        #     continue
        for dim in ndataset[var].dims:
            if dim_list is None or dim in dim_list:
                try:
                    ndataset[var] = xr.apply_ufunc(remove_numpy_dim, dataset[var], input_core_dims=[[dim]])
                except DimRemoveExcpt:
                    pass
                except Exception as e:
                    e.add_note(f"Problem while auto remove dim of variable={var}")
                    raise e
    return ndataset


def thread_vectorize(func, dim, max_workers=20, **kwargs):
    import concurrent
    new_args=[]


def resample_arr(a: xr.DataArray, dim: str, new_fs: float, position="centered", new_dim_name=None,*, mean_kwargs={}, return_counts=False):
    if new_dim_name is None:
        new_dim_name = f"{dim}_bins"
    match position:
        case "centered":
            a["new_fs_index"] = np.round(a[dim]*new_fs+1/(1000*new_fs))
        case "start":
            a["new_fs_index"] = (a[dim]*new_fs).astype(int)
    grp = a.groupby("new_fs_index")
    binned = grp.mean(dim, **mean_kwargs)
    binned = binned.rename(new_fs_index=new_dim_name)
    binned[new_dim_name] = binned[new_dim_name]/new_fs
    
    if return_counts:
        counts = grp.count(dim)
        counts= counts.rename(new_fs_index=new_dim_name)
        counts[new_dim_name] = counts[new_dim_name]/new_fs
        return binned, counts
    else:
        return binned
    
def sampled_arr_from_events(a: np.array, fs: float, weights=1):
    if len(a.shape) > 1:
        raise Exception(f"Wrong input shape. Got {a.shape}")
    m=np.round(np.min(a)*fs)
    M=np.round(np.max(a)*fs)
    n = int(M-m + 1)
    res = np.zeros(n)
    np.add.at(res, np.round(a*fs).astype(int) - int(m), weights)
    return res, m/fs

def sum_shifted(a: np.array, kernel: np.array):
    if len(a.shape) > 1:
        raise Exception(f"Wrong input shape. Got {a.shape}")
    if len(kernel.shape) > 1:
        raise Exception(f"Wrong input shape. Got {kernel.shape}")
    if kernel.size % 2 !=1:
        raise Exception(f"Kernel must have odd size {kernel.size}")
    roll = int(np.floor(kernel.size/2))
    a = np.concatenate([np.zeros(roll), a, np.zeros(roll)])
    # kernel = np.concatenate([kernel, np.zeros(a.size - kernel.size)])
    res = np.zeros(a.size)
    for i in range(-roll, roll+1):
        res = res + np.roll(a,i)*kernel[i+roll]
    return res

def normalize(a: xr.DataArray):
    std = a.std()
    a_normal = (a - a.mean()) / std
    return a_normal

def apply_file_func_decorator(base_folder, **kwargs):
    def decorator(f):
        def new_f(*arr_paths):
            return apply_file_func(f, base_folder, *arr_paths, **kwargs)
        return new_f
    return decorator

def extract_unique(a: xr.DataArray, dim: str):
    def get_uniq(a):
        nums = nunique(a, axis=-1, to_str=True)
        if (nums==1).all():
            r = np.take_along_axis(a, np.argmax(~pd.isna(a), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
            return r
        else:
            raise Exception(f"Can not extract unique value. Array:\n {a}\nExample\n{a[np.argmax(nums)]}")
    return xr.apply_ufunc(get_uniq, a, input_core_dims=[[dim]])


def mk_bins(a: xr.DataArray, dim, new_dim, coords, weights=None):
    # print(f"input arr sum {float(a.sum())}")
    if not weights is None:
        tmp = xr.apply_ufunc(lambda x, y: f"{x}_{y}", a, weights, vectorize=True)
    else:
        tmp = xr.apply_ufunc(lambda x: f"{x}_{1}", a, vectorize=True)
    weights = None
    # print(a)
    # print(weights)
    # exit()
    def compute(a: np.ndarray, weights=None):
        # if np.nansum(a) >0:
        #     print(f"compute input sum {np.nansum(a)}")
        # print(a)
        def make_hist(a: np.ndarray, weights=None):
            # print(f"make_hist input sum {np.nansum(a)}")
            # import time
            # time.sleep(0.01)
            tmp = np.array([list(x.split("_")) for x in a])
            a,weights = tmp[:, 0].astype(float), tmp[:, 1].astype(float)
            h, edges = np.histogram(a, coords, weights=weights)
            return h
        if weights is None:
            r = np.apply_along_axis(make_hist, axis=-1, arr= a)
        else:
            r = np.apply_along_multiple_axis(make_hist, axis=-1, arrs= [a, weights])
        # print(r)
        return r
    if weights is None:
        res: xr.DataArray = xr.apply_ufunc(compute, tmp, input_core_dims=[[dim]], output_core_dims=[[new_dim]])
    else:
        res: xr.DataArray = xr.apply_ufunc(compute, a, weights, input_core_dims=[[dim], [dim]], output_core_dims=[[new_dim]])
    # print(res)
    res = res.assign_coords({new_dim:(coords[1:]+ coords[:-1])/2, f"{new_dim}_low_edge": (new_dim, coords[:-1]), f"{new_dim}_high_edge": (new_dim, coords[1:])})
    return res


from typing import Any

class stupid:
        def __init__(self, grp, dataset):
            self.grp = grp
            self.dataset = dataset

        def __getattr__(self, __name: str) -> Any:
            f = self.grp.__getattribute__(__name)
            def new_f(*args, **kwargs):
                ret = f(*args, **kwargs)
                return xr.merge([ret, self.dataset])
            return new_f

def nicegroupby(self: xr.Dataset, val, *args, **kwargs):
    vars = [v for v in self.data_vars if set(self[val].dims).issubset(set(self[v].dims))]
    ret = stupid(self[vars].groupby(val, *args, **kwargs), self.drop_dims(self[val].dims))
    return ret



def add_postfix(p, d):
    if not hasattr(p, "addinfo"):
        p.addinfo = {}
    p.addinfo.update(d)
    p.set_postfix(p.addinfo)

def remove_postfix(p, l):
    if not hasattr(p, "addinfo"):
        return
    for k in l:
        p.addinfo.pop(k, None)
    p.set_postfix(p.addinfo)

def bootstrap(f, arr: xr.DataArray, sample_dim, n_resamples=10000,  bt_dist="bt_dist", vectorize=False, executor: concurrent.futures.Executor=None, _progress=None):
    import tqdm, tqdm.notebook, concurrent
    if arr.isnull().all():
        return xr.apply_ufunc(lambda a: np.full(a.shape[:-1] + (n_resamples,), np.nan), arr, input_core_dims=[[sample_dim]], output_core_dims=[["bt_dist"]])
    if vectorize is True:
        vectorize=n_resamples
    if vectorize is False:
        vectorize=1
    n = arr.sizes[sample_dim]

    def mk_sample(start, end):
        samples = np.arange(start, min(end, n_resamples))
        choices = np.random.choice(np.arange(n), len(samples) * n, replace=True).reshape((n, len(samples)))
        choices = xr.DataArray(data=choices, dims=["sample", bt_dist], coords={"sample":np.arange(n), bt_dist:samples})
        sample = arr.isel({sample_dim:choices}).drop([sample_dim]).rename(sample=sample_dim)
        return sample

    all_res = []
    if _progress is None or _progress is False:
        progress = lambda x: x
    elif _progress is True:
        progress = lambda x: tqdm.auto.tqdm(x, desc="Submitting", postfix=dict(vectorize=vectorize))
    else:
        def inner(x):
            add_postfix(_progress, dict(vectorize=vectorize))
            for i, it in enumerate(x):
                add_postfix(_progress, dict(block=f"{i}/{int(np.ceil(n_resamples/vectorize))}"))
                yield it
            remove_postfix(_progress, ["block", "vectorize"])
        progress = inner
    if not executor is None:
        futures = {}
        add_postfix(_progress, dict(action="submit"))
        for i in progress(range(0, n_resamples, vectorize)):
            # sample = mk_sample(i, i+vectorize)
            # if vectorize == 1:
            #     sample = sample.isel(sample_dim=0)
            futures[i] = executor.submit(lambda f,*args: f(mk_sample(*args)), f, i, i+vectorize)
        add_postfix(_progress, dict(action="wait"))
        for f in progress(concurrent.futures.as_completed(futures.values())):
            all_res.append(f.result())
    else:
        for i in progress(range(0, n_resamples, vectorize)):
                sample = mk_sample(i, i+vectorize)
                # if vectorize == 1:
                #     sample = sample.isel(sample_dim=0)
                all_res.append(f(sample))
    add_postfix(_progress, dict(action="concat"))
    final: xr.DataArray= xr.concat(all_res, dim=bt_dist)
    remove_postfix(_progress, ["action"])
    return final
    
def progress_map(self, f, *args, desc=None, **kwargs):
    import tqdm, tqdm.auto, inspect
    bar = tqdm.auto.tqdm(total=len(self.groups), desc=desc)
    k= list(self.groups.keys())
    i=0
    def new_f(*a, **kw):
        nonlocal i
        add_postfix(bar, dict(group=k[i]))
        bar.update(1)
        if "_progress" in inspect.getfullargspec(f).args:
            res= f(*a, _progress = bar, **kw)
        else:
            res= f(*a, **kw)
        i+=1
        return res
    res = self.map(new_f, *args, **kwargs)
    remove_postfix(bar, ["group"])
    bar.close()
    return res

def group_as_dim(var:str, data: xr.Dataset,  new_dim=None):
    if new_dim is None:
        new_dim = var
    v= data[var].to_numpy().flatten()
    v =v[~pd.isna(v)]
    values = np.unique(v)
    d = {str(v): data.where(data[var]==v).drop_vars([var]) for v in values}
    res = xr.concat(list(d.values()), dim=new_dim)
    res[new_dim] = list(d.keys())
    return res

import inspect

class UfuncApplier:
    in_dims: Dict[str, Sequence[str]] #Mapping from parameter names to function dimension
    out_dims: Sequence[Sequence[str]] #Mapping from output names to function dimension
    dims: Set[str] #All function dimensions
    dim_mapping: Dict[str, Sequence[str]] #Mapping from function dimension to calling array dimension

    def __init__(self, f, in_dims, out_dims, vectorized, array_type):
        self.f = f
        sig = inspect.signature(self.f).parameters
        if not hasattr(in_dims, "items"):
            in_dims = {str(p): in_dims[i] for i,p in enumerate(sig) if i<len(in_dims)}
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.vectorized = vectorized
        self.array_type = array_type
        self.dims = set().union(*([set(v) for v in in_dims.values()] + [set(v) for v in out_dims]))
        self.dim_mapping = {d:[d] for d in self.dims}

    def map_dims(self, dims=None, **dims_kwargs):
        dim_dict = dims_kwargs if dims is None else dims
        if set(dim_dict.keys()) - self.dims != set():
            raise Exception(f"Dimensions {set(dim_dict.keys()) - self.dims} are unknown")
        self.dim_mapping.update({k:[v] if isinstance(v, str) else v for k, v in dim_dict.items()})
        return self
    
    def __str__(self):
        sig = inspect.signature(self.f).parameters
        l = [f"{p}" if not p in self.in_dims else f"{p} : {self.in_dims[p]}" for p in sig]
        return f"{'Not vectorized' if not self.vectorized else 'Vectorized'} {self.f}({', '.join(l)})"
    
    def __call__(self, *args, **kwargs):
        sig = inspect.signature(self.f)
        arg_dict = sig.bind(*args, **kwargs).arguments
        kwname = [k for k in arg_dict if sig.parameters[k].kind ==inspect.Parameter.VAR_KEYWORD]
        posname = [k for k in arg_dict if sig.parameters[k].kind ==inspect.Parameter.VAR_POSITIONAL]
        if posname:
            posname = posname[0]
            posname_dict = {f"_pos{i}": v for i,v in enumerate(arg_dict[posname])}
            arg_dict.pop(posname)
            arg_dict = dict(**arg_dict, **posname_dict)
        else:
            posname_dict= {}
        
        if kwname:
            kwname = kwname[0]
            kwname_dict = arg_dict[kwname]
            arg_dict.pop(kwname)
            arg_dict = dict(**arg_dict, **kwname_dict)
        else:
            kwname_dict= {}
        
        arg_names = list(arg_dict.keys())
        def get_in_dims(k):
            if k in kwname_dict and kwname in self.in_dims:
                return self.in_dims[kwname]
            elif k in posname_dict and posname in self.in_dims:
                return self.in_dims[posname]
            elif k in self.in_dims:
                return self.in_dims[k]
            else:
                return []
            
        in_dims = {k: get_in_dims(k) for k in arg_names}
        positions = [k for (k, p) in sig.parameters.items() if p.kind==inspect.Parameter.POSITIONAL_OR_KEYWORD | p.kind==inspect.Parameter.POSITIONAL_ONLY]
        if posname:
            positions+=list(posname_dict.keys())
        input_core_dims = {k: [self.dim_mapping[d] for d in in_dims[k]] for k in arg_names}
        output_core_dims = {k: [self.dim_mapping[d] for d in l] for k, l in enumerate(self.out_dims)}

        
        def inner(*args):
            input_dimension_shapes = {}
            mapped_shape = {}
            for k, arg in zip(arg_names, args):
                arg_shape = np.shape(arg)
                input_dims_size = np.sum([len(self.dim_mapping[d]) for d in in_dims[k]]).astype(int)
                # print(arg_shape, input_dims_size)
                curr = len(arg_shape) - input_dims_size
                # print(curr)
                mapped_shape[k] = arg_shape[:curr]
                for d in in_dims[k]:
                    input_dimension_shapes[d] = arg_shape[curr: curr+len(self.dim_mapping[d])]
                    curr+=len(self.dim_mapping[d])

            # print(input_dimension_shapes)
            new_args={k: np.reshape(arg, mapped_shape[k] +tuple(np.prod(input_dimension_shapes[d]) for d in in_dims[k])) for k,arg in zip(arg_names, args)}
            posargs = [new_args[k] for k in positions]
            keywordargs = {k: v for k, v in new_args.items() if not k in positions}
            res = self.f(*posargs, **keywordargs)
            rm_tuple=False
            if not isinstance(res, tuple):
                res = (res,)
                rm_tuple=True
            output_shape = [() for _ in res]
            for i, r in enumerate(res):
                r_shape = np.shape(r)
                curr = len(r_shape) - len(self.out_dims[i])
                output_shape[i]+= r_shape[: curr]
                for d in enumerate(self.out_dims[i]):
                    output_shape[i]+=input_dimension_shapes[d] if d in input_dimension_shapes else (r_shape[curr],)
                    curr+=1
            # print([r.shape for r in res])
            new_res = tuple(np.reshape(r, output_shape[i]) for i, r in enumerate(res))
            # print([r.shape for r in new_res])
            # print(new_res)
            return new_res if not rm_tuple else new_res[0]

        res = xr.apply_ufunc(inner, *[arg_dict[k] for k in arg_names], 
            input_core_dims=[sum(input_core_dims[k], []) for k in arg_names],
            output_core_dims=[sum(output_core_dims[k], []) for k in output_core_dims],
            vectorize=not self.vectorized,
        )


        
        
        return res
    
    def exec(self, *args, **kwargs): return self.__call__(*args, **kwargs)

class tmp:
    def __init__(self, f, in_dims, out_dims, vectorized, array_type):
        self.f = f
        self.xr = UfuncApplier(f, in_dims, out_dims, vectorized, array_type)
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

def xrfunc(in_dims={}, out_dims=[[]], vectorized=False, array_type="numpy"):
    def decorator(f):
        return tmp(f, in_dims, out_dims, vectorized, array_type)
    return decorator

def oldxrfunc(in_dims={}, out_dims=[[]], vectorized=False, array_type="numpy"):
    in_dimension_list=[d for v in in_dims.values() for d in v]
    def decorator(f):
        def new_f(*args, dims=None,  ufunc_kwargs=None, **kwargs):
            dim_dict = kwargs if dims is None else dims
            in_dims_mapping = {d: [d] if not d in dim_dict else [dim_dict[d]] if isinstance(dim_dict, str) else dim_dict[d]}
            removed_dict_keys = [k for k in d for d in [dims, ufunc_kwargs] if not d is None]
            fkwargs = {k:v for k,v in kwargs.items() if k not in removed_dict_keys}

            arg_dict = inspect.signature(f).bind(*args, **fkwargs).arguments
            arg_names = list(arg_dict.keys())


            input_core_dims = {k:[] for k in arg_names}
            reshape_sizes = {k:[] for k in arg_names}
            for k in arg_names:
                if k in in_dims:
                    mdim = in_dims[k]
                    for d in mdim:
                        if d in kwargs:
                            if isinstance(kwargs[d], str):
                                input_core_dims[k].append(kwargs[d])
                                reshape_sizes[k].append(1)
                            else:
                                # print(input_core_dims[k])
                                input_core_dims[k]+=kwargs[d]
                                # print(input_core_dims[k])
                                reshape_sizes[k].append(len(kwargs[d]))
                        else:
                            raise Exception("Not handled yet")
            # input_core_dims = {k: ([kwargs[d] for d in in_dims[k]] if k in in_dims else []) for k in arg_dict}
            # input_core_dims ={k}
            
            def inner(*args): 
                # print("inner !")
                new_args =[]
                for i, arg in enumerate(args):
                    try:
                        if len(np.shape(arg))==0:
                            new_args.append(arg)
                            continue
                        k = arg_names[i]
                        reshapes = reshape_sizes[k]
                        total = np.sum(reshapes)
                        arg_shape = np.shape(arg)
                        start = len(arg_shape) - total
                        new_shape = [s for i,s in enumerate(arg_shape) if i < start] 
                        curr = start
                        for curr_reshape in range(len(reshapes)):
                            new_shape.append(np.prod(arg_shape[curr:curr+reshapes[curr_reshape]]))
                            curr+=reshapes[curr_reshape]
                        new_args.append(np.reshape(arg, new_shape))
                    except:
                        print(type(arg))
                        raise
                ret = f(*new_args)
                return ret
            # print(f"Input_core_dims = {input_core_dims}")
            # print(*[arg_dict[k] for k in arg_names])
            return xr.apply_ufunc(inner, *[arg_dict[k] for k in arg_names], input_core_dims=[input_core_dims[k] for k in arg_names], vectorize = not vectorized)
        return new_f
    return decorator



# @xrfunc(vectorize=False, in_dims=dict(a=["samples", "features"]), out_dims=dict(pca = ["samples", "pca_features"]))
# def pca(a, *args):
#     pass

# pca(a, samples=["t", "x"], features="g")

# def apply_xrfunc(func, *args, vectorize="auto", join="exact", **kwargs):
#     if hasattr(func, "_xrarray_ufunc"):



    # new_input_core_dims = [[d for d in input_core_dims if hasattr(arg, "dims") and d in arg.dims] for arg in args]
    # if hasattr(func, "_supports_vectorization")