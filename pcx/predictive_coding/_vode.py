__all__ = [
    "STATUS",
    "Ruleset",
    "Vode",
]

import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import jax

from ..core._module import BaseModule
from ..core._parameter import Param
from ..core._random import RKG, RandomKeyGenerator
from ..core._static import static
from ._energy import se_energy
from ._energy_module import EnergyModule
from ._parameter import VodeParam

########################################################################################################################
#
# VODE
#
# Vode is the fundamental building block to transform a deep learning network into a predictive coding network by
# breaking the continous flow of information from input to output layer into independent, stateful blocks.
# The standard usage is 'x = vode(act_fn(layer(x)))'. The behaviour of a Vode can be customised by specifying the
# 'energy_fn' and its 'ruleset'.
#
########################################################################################################################

# Core #################################################################################################################


class STATUS:
    """
    List of common statuses used in predictive coding networks. Any string can be used as a status, but these
    are the most common ones. The 'STATUS.INIT' status is used to forward initialise the Vode value 'h' with the
    incoming activation 'u' in the default ruleset.
    """

    NONE = None
    ALL = ".*"
    INIT = "init"


class Ruleset(BaseModule):
    """
    A Ruleset transform input and output values of a vode according to the specified rules. A set of rules can be
    specified as a tuple of either input (i.e., 'target <- key:transformation') or output (i.e.,
    'key -> target:transformation') rules. Each set of rules is associated to a set of statuses that matches the
    current status of the Vode. The matching is determined by the regular expression pattern specified for each set
    of rules (i.e., '.*': (rule1, rule2) would  apply the two rules to any status). If multiple input rules match
    the current status and operation, the are all executed in the order they are specified. If multiple output rules
    match the current status and operation, only the first one is executed.
    """

    def __init__(
        self,
        rules: dict[str, Sequence[str]],
        tforms: dict[
            str,
            Callable[["Vode", str, jax.Array | None, RandomKeyGenerator], jax.Array | None],
        ] = {},
    ):
        """Ruleset constructor.

        Args:
            rules (Dict[str, Sequence[str]]): dictionary of set of rules, where each key is a regular expression to
                match the current status of the Vode, and each value is a sequence of rules to apply.
            tforms (Dict[str, Callable[['Vode', str, jax.Array | None, RandomKeyGenerator], jax.Array | None]]],
                optional): custom transformations provided to the ruleset.
        """
        super().__init__()

        self.rules = static(rules)
        self.tforms = static(tforms)

    def filter(self, status: str | None, rule_pattern: str):
        """Filter all the rules that match the current status and the given rule pattern.

        Args:
            status (str | None): the target status to match.
            rule_pattern (str): the target rule pattern to match.

        Yields:
            Tuple[str, str]: the target and transformation of the rule.
        """
        status = status or ""

        for _pattern, _rules in self.rules.items():
            if re.fullmatch(_pattern, status) is None:
                continue

            for _rule in _rules:
                if _match := re.fullmatch(rule_pattern, _rule):
                    yield _match.group(1, 2)

    def apply_set_transformation(
        self,
        node: "Vode",
        tform: str,
        key: str,
        value: Any | None = None,
        rkg: RandomKeyGenerator = RKG,
    ) -> Any | None:
        """Recursively apply the transformation specified by the given tform to the given value.

        Args:
            node (Vode): target vode.
            tform (str): sequence of ":"-separated transformations to apply to the value.
            key (str): input key.
            value (Any | None, optional): input value.
            rkg (RandomKeyGenerator, optional): random key generator. Defaults to RKG.

        Returns:
            Any | None: transformed value.
        """

        if ":" in tform:
            tform, _t = tform.rsplit(":", 1)

            value = self.tforms[_t](
                node,
                key,
                self.apply_set_transformation(node, tform, key, value, rkg),
                rkg,
            )

        return value

    def apply_get_transformation(self, node: "Vode", tform: str, key: str, rkg: RandomKeyGenerator = RKG) -> Any | None:
        """Recursively apply the transformation specified by the given tform to the given value.

        Args:
            node (Vode): target vode.
            tform (str): sequence of ":"-separated transformations to apply to the value.
            key (str): input key.
            value (Any | None, optional): input value.
            rkg (RandomKeyGenerator, optional): random key generator. Defaults to RKG.

        Returns:
            Any | None: transformed value.
        """
        _value = node.get(tform, None)

        if _value is None and ":" in tform:
            tform, _t = tform.rsplit(":", 1)

            _value = self.tforms[_t](node, key, self.apply_get_transformation(node, tform, key, rkg), rkg)

        return _value


class Vode(EnergyModule):
    """
    Base and configurable class for Vectorised Nodes. In a predictive coding network, a Vode is any element whose
    state depends on a particular sample data provided to the network (in contrast with the module weights, which
    are shared across all samples). In predictive coding, the most common type of Vode is a value node, usually
    denoted with 'x' in the literature. A value node is characterised by an energy function, which calculates the
    error between the predicted and the actual value of the node.

    A Vode offers a series of methods to customise its behavioir, with its default configuration being the one used
    by Gaussian value nodes. The user can define a custom energy function, a custom set of rules to update the Vode
    and simply inherits from it do define even more customised behaviour."""

    def __init__(
        self,
        energy_fn: Callable[["Vode", RandomKeyGenerator], jax.Array] = se_energy,
        ruleset: dict = {},
        tforms: dict = {},
        param_type: type[VodeParam] = VodeParam,
        *param_args,
        **param_kwargs,
    ):
        """Vode constructor.

        Args:
            energy_fn (Callable[['Vode', RandomKeyGenerator], jax.Array], optional): function used to compute the Vode
                energy.
            axes (int): number of dimensions (not including the batch dimension) of the Vode value. It should match
                the input activation 'u'. It is used to determine wether the Vode is being used in a vmapped context.
                The default value None assumes that the first activation received by the Vode is vmapped over the batch
                dimension and sets the number of axes accordingly.
            ruleset (Ruleset, optional): ruleset specifying the Vode behaviour. The default value indicates that, with
                status set to 'STATUS.INIT', the incoming activation 'u' is also saved to the value 'h', which
                corresponds to forward initialisation.
            param_type (type[VodeParam], optional): the parameter type of the value 'h'. Defaults to VodeParam.
            *param_args, **param_kwargs: arguments passed to the 'param_type' constructor.
        """
        super().__init__()

        self.h = param_type(*param_args, **param_kwargs)
        self.cache = param_type.Cache()
        self.energy_fn = static(energy_fn)
        self.ruleset = Ruleset({STATUS.INIT: ("h, u <- u",), **ruleset}, tforms)
        self.shape = static(None)

    def __call__(self, u: jax.Array | None, rkg: RandomKeyGenerator = RKG, output="h", **kwargs) -> jax.Array | Any:
        """Deep learning layers are typically implemented as callable objects, taking in input the incoming activation
        and returning the transformed activation. Analogously, a Vode is implemented as a callable object, taking in
        input the Vode incoming activations (e.g., 'u' and/or other values), storing them, and returning the Vode value
        'h' (the output can be customised by setting the 'output' parameter to the desired value).
        __call__ is equivalent to 'vode.set("u", u).get("h")'.

        Args:
            u (jax.Array | None): if provided, it sets the incoming activation 'u' to the given value.
            rkg (RandomKeyGenerator, optional): random key generator. Defaults to RKG.
            output (str, optional): Value to return. Defaults to "h". If 'None', the Vode object is returned.
            **kwargs: optional additional activations to set.

        Returns:
            jax.Array | 'Vode': output value corresponding to the selected output parameter.
        """

        if u is not None:
            self.set("u", u, rkg)

        for _k, _v in kwargs.items():
            self.set(_k, _v, rkg)

        if (h := self.get("u")) is not None:
            # Record the input shape to distinguish between vmapped and non-vmapped contexts when calling .energy
            self.shape.set(h.shape)

        if output is None:
            return self
        else:
            return self.get(output, rkg=rkg)

    def set(self, key: str, value: jax.Array | None, rkg: RandomKeyGenerator = RKG) -> "Vode":
        """Set the value of the parameter corresponding to the given key, after being processed by the Vode ruleset.
        The rule syntax is 'target <- key:transformation', where 'target' is the name of the parameter to set (can be a
        list of comma-separated names), and 'transformation' is a string that refers to the name of the transformation
        to apply to 'value' before saving it to the target. The transformation must be a method provided to the ruleset
        at construction time. A transformation signature is
        'def transformation(vode: Vode, key: str, value: jax.Array | None, rkg: RandomKeyGenerator) -> jax.Array | None'
        Transformations can be chained (e.g., 'h <- u:se:zero').

        Args:
            key (str): name of the parameter to set. If the parameter is not found, it is stored in the cache.
            value (jax.Array | None): value to set.
            rkg (RandomKeyGenerator, optional): random key generator. Defaults to RKG.

        Returns:
            Vode: returns itself to allow for chaining.
        """
        # The key group is anchored so 'u' cannot match a rule written for 'u2'; the optional tail is the
        # ':transformation' suffix. 'Ruleset.filter' full-matches, so no trailing anchor is needed here.
        _rule_pattern = f"(.*(?<!\\s))\\s*<-\\s*({key}(?::.*)?)"

        rules = tuple(self.ruleset.filter(self.status, _rule_pattern))
        for _targets, _tform in rules:
            _value = self.ruleset.apply_set_transformation(self, _tform, _tform.split(":", 1)[0], value, rkg)

            for _target in _targets.split(","):
                _target = _target.strip()
                if hasattr(self, _target) and isinstance((_param := getattr(self, _target)), Param):
                    _param.set(_value)
                else:
                    self.cache[_target] = _value

        if len(rules) == 0:
            if hasattr(self, key) and isinstance((_param := getattr(self, key)), Param):
                _param.set(value)
            else:
                self.cache[key] = value

        return self

    def get(self, key: str, default: Any | None = None, rkg: RandomKeyGenerator = RKG) -> jax.Array | Any | None:
        """Returns the value of the parameter corresponding to the given key, after being processed by the Vode ruleset.
        The rule syntax is 'key -> target:transformation', where 'target' is the name of the parameter to get when
        key is queried. NOTE: the right-hand side of the rule is also saved to the cache, so subsequent calls to the
        same key will return the same value without recomputation.

        Args:
            key (str): name of the parameter to get.
            default (Any | None, optional): default value to return if the parameter is not found.
            rkg (RandomKeyGenerator, optional): random key generator. Defaults to RKG.

        Returns:
            jax.Array | Any | None: the value of the parameter corresponding to the given key.
        """
        _rule_pattern = f"({key})\\s*->\\s*(.*)"

        _rules = tuple(self.ruleset.filter(self.status, _rule_pattern))

        if len(_rules) == 0:
            if hasattr(self, key) and isinstance((_param := getattr(self, key)), Param):
                return _param.get()
            else:
                return self.cache.get(key, default)
        else:
            if len(_rules) > 1:
                warnings.warn(
                    f"Multiple output rules matched for key '{key}' in status '{self.status}', "
                    "only the first is applied.",
                    stacklevel=2,
                )
            (_target, _tform) = _rules[0]

            _value = self.ruleset.apply_get_transformation(self, _tform, _target, rkg=rkg)

            if ":" in _tform:
                self.cache[_tform] = _value

            return _value

    def energy(self, rkg: RandomKeyGenerator = RKG) -> jax.Array:
        """Compute the Vode energy and saves it to the cache, using the key 'E'.
        The energy is computed by the energy function provided at construction time.
        Information about individual samples is preserved and the energy is returned as a vector
        with shape (batch_size,).

        Args:
            rkg (RandomKeyGenerator, optional): random key generator. Defaults to RKG.

        Returns:
            jax.Array: Vode energy
        """
        if "E" not in self.cache:
            if self.energy_fn.get() is not None:
                _E = self.energy_fn(self, rkg=rkg)

                if self.h.shape == self.shape:
                    # if the shape is the same as the vode shape,
                    # '.energy' is being called from a vmapped function
                    # otherwise 'h' would have a an extra dimension (batch)
                    _E = _E.sum()
                else:
                    # .energy is being called from a non-vmapped function
                    # we want to preserve the energy information of each element
                    _E = jax.numpy.reshape(_E, (self.h.shape[0], -1)).sum(axis=1)

                self.cache["E"] = _E
            else:
                return 0.0

        return self.cache["E"]
