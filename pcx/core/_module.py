__all__ = ["BaseModule", "Module"]


import abc
import functools
from collections.abc import Generator
from enum import IntEnum
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.tree_util as jtu

from ._parameter import DynamicParam
from ._static import static
from ._tree import tree_apply

T = TypeVar("T")


########################################################################################################################
#
# MODULE
#
# as in equinox, each Module is a JAX pytree, and can be used as a container for other Modules and Parameters.
# Thus, each class that inherits from Module is automatically registered as a JAX pytree whose children are the
# attributes of the class. This allows to flatten and unflatten the class as if it were a dictionary.
# For example:
#
# ```python
# class Obj(Module):
#    def __init__(self, x, y):
#       self.x = x
#       self.y = y
#
# obj = Obj(1, 2)
# leaves, structure = jtu.tree_flatten(obj)
# print(leaves)  # (1, 2), structure contains the keys of the attributes "x" and "y"
# ```
########################################################################################################################


# Core #################################################################################################################


class _BaseModuleMeta(abc.ABCMeta):
    """
    Metaclass to register all modules as JAX pytrees so that can be (un)flattened.
    A module is flattened as if it were a dictionary, separating keys and values.

    NOTE: as equinox does, it may be necessary to update this code to treat __special__ attributes differently.
    """

    def __new__(mcs, name, bases, dct):
        _cls = super().__new__(mcs, name, bases, dct)

        jax.tree_util.register_pytree_with_keys(
            _cls,
            flatten_func=_BaseModuleMeta.flatten_module,
            flatten_with_keys=_BaseModuleMeta.flatten_module_with_keys,
            unflatten_func=functools.partial(_BaseModuleMeta.unflatten_module, cls=_cls),
        )

        return _cls

    @staticmethod
    def flatten_module(module: "BaseModule") -> tuple[tuple[Any, ...], tuple[str, ...]]:
        # Keys are sorted, as jax does for dictionaries, so that the structure does not depend on assignment order.
        _keys = tuple(sorted(module.__dict__))

        return tuple(module.__dict__[k] for k in _keys), _keys

    @staticmethod
    def flatten_module_with_keys(
        module: "BaseModule",
    ) -> tuple[tuple[tuple[str, Any], ...], tuple[str, ...]]:
        _keys = tuple(sorted(module.__dict__))

        return (
            tuple((jtu.GetAttrKey(k), module.__dict__[k]) for k in _keys),
            _keys,
        )

    @staticmethod
    def unflatten_module(aux_data: tuple[str, ...], children: tuple[Any, ...], cls: type["BaseModule"]) -> "BaseModule":
        _module = object.__new__(cls)

        _module.__dict__ = dict(zip(aux_data, children, strict=True))

        return _module


class BaseModule(metaclass=_BaseModuleMeta):
    """
    _BaseModule is the base class for all modules in the library.
    """

    def __call__(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        leaves = jtu.tree_leaves_with_path(self, is_leaf=lambda x: isinstance(x, DynamicParam))

        if len(leaves):
            return "\n".join(
                (f"({self.__class__.__name__}):", *(f"  {jtu.keystr(key)}: {value!r}" for key, value in leaves))
            )
        else:
            return f"({self.__class__.__name__}):  (empty)"

    def submodules(self, *, cls: type[T] | None = None) -> Generator[T, None, None]:
        """Return the children submodules of the given type. Does not work recursively, and
        only returns the direct children of matching type.

        Args:
            cls (Type[T] | None, optional): indicates the type of the submodules to select.
                If None, '_BaseModule' is used.

        Yields:
            Generator[T, None, None]: genereator of the matched submodules.
        """
        # We target all modules by default.
        cls = cls or BaseModule
        _leaves, _ = eqx.tree_flatten_one_level(self)

        yield from filter(
            lambda x: isinstance(x, cls),
            jtu.tree_leaves(_leaves, is_leaf=lambda x: isinstance(x, cls)),
        )


# Module ###############################################################################################################


class Module(BaseModule):
    """
    Module represents a standard deep learning module with a train/eval mode flag that can be recursively set.
    """

    class MODE(IntEnum):
        NONE = 0
        TRAIN = 1
        EVAL = 2

    def __init__(self) -> None:
        self._mode = static(None)

    def mode(self, value: MODE | None) -> MODE | None:
        """Recursively set the mode of the module and its submodules.
        If the value is None, the current mode is instead returned.

        Args:
            value (MODE | None): mode to set.

        Returns:
            MODE | None: current mode if value is None, otherwise None.
        """
        if value is None:
            return self._mode.get()
        else:
            tree_apply(lambda m: m._mode.set(value), lambda x: isinstance(x, Module), self)

            return

    def train(self) -> None:
        """Set the module in train mode."""
        self.mode(Module.MODE.TRAIN)

        # Set equinox inference mode to False. Note that all attributes of an equinox module are automatically
        # converted to parameters, either static or dynamic, so we can access them via `.set()`/`.get()`.
        tree_apply(
            lambda eqx_m: eqx_m.inference.set(False),
            lambda x: hasattr(x, "inference"),
            self,
            False,
        )

    def eval(self) -> None:
        """Set the module in eval mode."""
        self.mode(Module.MODE.EVAL)

        # Set equinox inference mode to True.
        tree_apply(
            lambda eqx_m: eqx_m.inference.set(True),
            lambda x: hasattr(x, "inference"),
            self,
            False,
        )

    @property
    def is_train(self) -> bool:
        """Returns:
        bool: whether the module is in train mode.
        """
        return self._mode.get() == Module.MODE.TRAIN

    @property
    def is_eval(self) -> bool:
        """Returns:
        bool: whether the module is in eval mode.
        """
        return self._mode.get() == Module.MODE.EVAL
