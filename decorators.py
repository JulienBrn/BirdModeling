from typing import List, Tuple, Dict, Any, Literal, Set, Sequence, Callable, Mapping
import inspect

def simplify_decorators(f: Callable, *args, **kwargs):
    sig = inspect.signature(f)
    parameters = sig.parameters
    bind = sig.bind(*args, **kwargs)
    arg_dict = bind.arguments

    positional_mapping = {k:arg_dict[k] for k, p in parameters.items() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD | inspect.Parameter.POSITIONAL_ONLY}
    positional_args = tuple(positional_mapping.values()) + bind.args

    keyword_args = {k: arg_dict[k] for k, p in parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    new_args = positional_args + tuple(keyword_args.values()) + tuple(bind.kwargs.values())

    kwname = [k for k in parameters if parameters[k].kind ==inspect.Parameter.VAR_KEYWORD]
    posname = [k for k in parameters if parameters[k].kind ==inspect.Parameter.VAR_POSITIONAL]

    def new_f(*args):
        pos_args = args[:len(positional_args)]
        keyword_args = {k:arg for k,arg in zip(tuple(keyword_args.keys()) + tuple(bind.kwargs.keys()) , args[len(positional_args):])} 
        return f(*pos_args, **keyword_args)
    

    def transform_mapping(m: Mapping[str, Any], missing_value):
        positional_map = [m[k] if k in m else missing_value for k in positional_mapping]
        var_pos_map = ([m[posname] if posname in m else missing_value] * len(bind.args)) if posname else []
        keyword_map = [m[k] if k in m else missing_value for k in keyword_args.keys()]
        var_keyword_map = ([m[kwname] if kwname in m else missing_value] * len(bind.kwargs)) if kwname else []
        return positional_map + var_pos_map + keyword_map + var_keyword_map

    return new_f, new_args, transform_mapping
    if not _mappings is None:
        new_mappings = [transform_mapping(m, mv) for m, mv in zip(_mappings, _missing_values)]
        return new_f, new_args, new_mappings
    else:
        return new_f, new_args
    


class AnnotatedFunction:
    f: Callable
    input_info: Mapping[str, Tuple[Mapping[str, Any], Any]]
    output_info: Mapping[str, Tuple[Mapping[str, Any], Any]]
    output_names: Sequence[str] | None
    other_info: Dict[str, Any]

    def __init__(self, f):
        self.f = f
        self.input_info = {}
        self.output_info = {}
        self.output_names = None
        self.other_info = {}

    def __call__(self, *args, **kwargs):
        self.f(*args, **kwargs)

    def transform_inputs(self, *args, **kwargs):
        new_f, new_args, trs = simplify_decorators(self.f, *args, **kwargs)
        return new_f, new_args, {k: }

    def get_input_map(self, name):
        new_f, new_args, new_map = simplify_decorators(self.f, )

def decorate_inputs(f: Callable, mappings):
    import pandas as pd
    if not isinstance(f, AnnotatedFunction):
        f = AnnotatedFunction(f)
    if "inputs" not in f.info:
        f.info["inputs"] ={}
    f.info["inputs"].update(mappings)

def process(f: AnnotatedFunction): pass

@to_xr_func()
@add_vectorization(a=True, b=False)
@add_core_dim_names(a=None, b=["t"])
@add_input_types(a=int)
def func(a, b):pass