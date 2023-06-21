from typing import Callable, List, Dict, Optional, Any
import inspect
import jax


def move(obj: Any, target: Optional[Any] = None):
    return obj.__move__(target) if target is not obj else obj


def kwargs_indices(f: Callable, kwargs: Dict) -> List[int]:
    """Returns the indices of the keyword arguments of a function."""
    return [
        i
        for i, p in enumerate(inspect.signature(f).parameters.values())
        if p.name in kwargs
    ]


def hash_pytree(pytree: Any) -> int:
    leaves, treedef = jax.tree_util.tree_flatten(pytree)

    hashable_leaves = tuple((x.shape, x.dtype) if isinstance(x, jax.Array) else x for x in leaves)

    return hash((hashable_leaves, treedef))


def make_args(f: Callable, args=(), kwargs={}) -> List[str]:
    args_list = []
    args_it = iter(args)

    for parameter in inspect.signature(f).parameters.values():
        if parameter.name in kwargs:
            args_list.append(kwargs[parameter.name])
        else:
            try:
                args_list.append(next(args_it))
            except StopIteration:
                args_list.append(parameter.default)

    return args_list


def repr_function(f: Callable) -> str:
    """Human readable function representation."""
    signature = inspect.signature(f)
    args = [f'{k}={v.default}' for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty]
    args = ', '.join(args)
    while not hasattr(f, '__name__'):
        if not hasattr(f, 'func'):
            break
        f = f.func
    if not hasattr(f, '__name__') and hasattr(f, '__class__'):
        return f.__class__.__name__
    if args:
        return f'{f.__name__}(*, {args})'
    return f.__name__
