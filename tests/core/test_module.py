"""Contracts of `pcx.Module` / `pcx.BaseModule`.

A module is a pytree whose children are its attributes, so jax rebuilds it at
every transform boundary. Two properties follow, and the library leans on both:

* rebuilding must produce a *new* module holding the *same* parameter objects.
  `pcx.nn.shared` is nothing but a flatten/unflatten with parameters treated as
  leaves — if identity were lost there, "shared" weights would be copies that
  drift apart after the first update.
* the treedef must depend on what a module holds, not on the order in which the
  constructor happened to assign it, because a treedef mismatch turns an
  ordinary `tree_map` between two instances into an error.

Mode propagation and `submodules` are the two recursive traversals every layer
and energy computation is routed through.
"""

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from conftest import assert_allclose, count_leaves

import pcx


class Leafy(pcx.Module):
    """A minimal module: two parameters and one static."""

    def __init__(self, n: int = 3):
        super().__init__()
        self.w = pcx.Param(jnp.ones(n))
        self.b = pcx.Param(jnp.zeros(n))
        self.name = pcx.static("leafy")


class Nested(pcx.Module):
    """A module holding a direct child and a list of children."""

    def __init__(self):
        super().__init__()
        self.child = Leafy()
        self.children = [Leafy(), Leafy()]


def _one_level_roundtrip(module):
    """Flatten with parameters as leaves and rebuild — exactly what `shared` does."""
    leaves, treedef = jtu.tree_flatten(module, is_leaf=lambda x: isinstance(x, pcx.BaseParam))
    return jtu.tree_unflatten(treedef, leaves)


def test_flatten_unflatten_returns_a_new_module_object():
    """The rebuilt module must be a distinct object, or `shared` could not create
    a second module at all and mutating one would rewrite the other's structure."""
    m = Leafy()

    assert _one_level_roundtrip(m) is not m


def test_flatten_unflatten_preserves_parameter_identity():
    """THE property `pcx.nn.shared` is built on: with parameters treated as
    leaves, the rebuilt module must hold the very same `Param` objects.

    If these were copies, two "shared" layers would each own their own weights;
    training would update them independently and the model would quietly stop
    being weight-tied — with no error anywhere.
    """
    m = Leafy()

    rebuilt = _one_level_roundtrip(m)

    assert rebuilt.w is m.w
    assert rebuilt.b is m.b


def test_flatten_unflatten_preserves_static_identity():
    """Statics are leaves under the same filter, so they must be shared too —
    a copied static would let two supposedly identical layers drift in
    configuration."""
    m = Leafy()

    assert _one_level_roundtrip(m).name is m.name


def test_flatten_unflatten_preserves_the_concrete_class():
    """Every mask and filter in the library dispatches on module type; a rebuild
    that returned a base `Module` would make them all miss."""
    m = Leafy()

    assert type(_one_level_roundtrip(m)) is Leafy


def test_nested_modules_survive_a_one_level_roundtrip():
    """Sharing has to work through containers, since real models hold their
    layers in lists."""
    m = Nested()

    rebuilt = _one_level_roundtrip(m)

    assert rebuilt.child.w is m.child.w
    assert rebuilt.children[1].b is m.children[1].b


def test_full_roundtrip_through_jax_preserves_values():
    """A full flatten decomposes parameters into their arrays — the form in which
    a module crosses a transform boundary. Values must come back unchanged."""
    m = Leafy()
    m.w.set(jnp.array([1.0, 2.0, 3.0]))

    rebuilt = jtu.tree_unflatten(*jtu.tree_flatten(m)[::-1])

    assert type(rebuilt) is Leafy
    assert_allclose(rebuilt.w.get(), jnp.array([1.0, 2.0, 3.0]))
    assert_allclose(rebuilt.b.get(), jnp.zeros(3))


def test_module_leaf_count_is_the_number_of_dynamic_parameters():
    """Statics and the mode flag must not inflate the leaf count: that count is
    the size of every gradient and optimiser-state tree built from the module."""
    assert count_leaves(Leafy()) == 2
    assert count_leaves(Nested()) == 6


def test_two_instances_of_the_same_class_share_a_treedef():
    """Optimisers `tree_map` a gradient tree against a parameter tree, and jit
    caches on the treedef. Identically built instances must be interchangeable."""
    assert jtu.tree_structure(Leafy()) == jtu.tree_structure(Leafy())


def test_tree_map_across_two_instances_works():
    """The concrete consequence of the property above — this is a weight update
    in miniature."""
    summed = jtu.tree_map(lambda a, b: a + b, Leafy(), Leafy())

    assert_allclose(summed.w.get(), jnp.full((3,), 2.0))


def test_attribute_assignment_order_does_not_change_the_treedef():
    """A module is documented to be "flattened as if it were a dictionary", and a
    dictionary in jax is order-insensitive — `tree_structure({"a": 1, "b": 2})`
    equals `tree_structure({"b": 2, "a": 1})`, precisely so that structure depends
    on content rather than on insertion history.

    Modules instead key on `__dict__` order, so two instances of the same class
    whose constructor took different branches cannot be `tree_map`ed together and
    force a jit recompile, with an error that points at pytree internals rather
    than at the constructor.
    """

    class Branching(pcx.Module):
        def __init__(self, a_first: bool):
            super().__init__()
            if a_first:
                self.a = pcx.Param(jnp.ones(1))
                self.b = pcx.Param(jnp.zeros(1))
            else:
                self.b = pcx.Param(jnp.zeros(1))
                self.a = pcx.Param(jnp.ones(1))

    assert jtu.tree_structure(Branching(True)) == jtu.tree_structure(Branching(False))


def test_a_fresh_module_is_neither_train_nor_eval():
    """The mode starts unset so that a layer whose behaviour differs between the
    two (dropout, batch norm) cannot silently pick one by default."""
    m = Leafy()

    assert m.mode(None) is None
    assert m.is_train is False
    assert m.is_eval is False


@pytest.mark.parametrize(
    ("method", "expected"),
    [("train", pcx.Module.MODE.TRAIN), ("eval", pcx.Module.MODE.EVAL)],
    ids=["train", "eval"],
)
def test_mode_setters_report_through_mode_and_the_flags(method, expected):
    m = Leafy()

    getattr(m, method)()

    assert m.mode(None) == expected
    assert m.is_train is (expected is pcx.Module.MODE.TRAIN)
    assert m.is_eval is (expected is pcx.Module.MODE.EVAL)


@pytest.mark.parametrize("method", ["train", "eval"])
def test_mode_propagates_to_every_nested_module(method):
    """Calling `.eval()` on the top-level model must reach every layer, including
    ones held inside plain Python containers. A layer left in train mode would
    keep applying dropout at evaluation time and quietly degrade the reported
    metric."""
    m = Nested()

    getattr(m, method)()

    is_expected = (lambda x: x.is_train) if method == "train" else (lambda x: x.is_eval)
    assert is_expected(m)
    assert is_expected(m.child)
    assert all(is_expected(c) for c in m.children)


def test_switching_mode_clears_the_previous_mode_recursively():
    """Modes are mutually exclusive; a stale flag on a nested module would make
    `is_train` and `is_eval` disagree between parent and child."""
    m = Nested()
    m.train()

    m.eval()

    assert m.child.is_eval and not m.child.is_train
    assert all(c.is_eval and not c.is_train for c in m.children)


def test_submodules_returns_direct_children():
    m = Nested()

    found = list(m.submodules())

    assert len(found) == 3
    assert all(isinstance(x, Leafy) for x in found)


def test_submodules_does_not_include_the_module_itself():
    """`EnergyModule.energy` sums over `self.submodules()`; if `self` were
    included the recursion would never terminate."""
    m = Nested()

    assert all(x is not m for x in m.submodules())


def test_submodules_is_not_recursive():
    """Documented: "Does not work recursively, and only returns the direct
    children of matching type." Recursion is the caller's job — `energy()`
    recurses by calling `energy()` on each child, and a traversal that also
    descended would count every grandchild twice."""

    class Outer(pcx.Module):
        def __init__(self):
            super().__init__()
            self.inner = Nested()

    found = list(Outer().submodules())

    assert len(found) == 1
    assert isinstance(found[0], Nested)


def test_submodules_filters_by_class():
    """The `cls` filter is how `energy()` selects only energy-bearing children."""

    class Other(pcx.Module):
        pass

    class Mixed(pcx.Module):
        def __init__(self):
            super().__init__()
            self.a = Leafy()
            self.b = Other()

    m = Mixed()

    assert [type(x) for x in m.submodules(cls=Other)] == [Other]
    assert [type(x) for x in m.submodules(cls=Leafy)] == [Leafy]


# A module reachable through two attributes is yielded twice, so a tied Vode
# contributes its energy twice to `EnergyModule.energy`. That is out of contract
# rather than a defect: the library is designed on the assumption that a module
# is reachable exactly once, so the behaviour is undefined and nothing is
# asserted about it here. See BUGS.md for the open question about documenting
# the precondition and about `tree_ref`, which does deduplicate on `id()`.
# Shared *parameters* are a different matter and are supported: see
# `tests/core/test_tree.py` and issue #73.
