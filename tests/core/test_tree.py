"""Contracts of the ref/unref and extract/inject machinery.

jax only accepts pytrees: a *tree*, where every node appears once. pcx models are
pydags — the same `Param` object is deliberately reachable by several paths when
weights are tied. `tree_ref` bridges the two by rewriting every repeat occurrence
as a static index, and `tree_unref` puts the original object back.

That round-trip is the load-bearing property of the whole library. If it lost
identity, a tied weight would arrive inside a transform as two independent leaves:
it would receive two separate gradients instead of their sum, be updated twice per
step, and come back out as two objects that immediately drift apart — all without
raising anything.

Expectations here come from what "the same object, referenced twice" means and
from the documented behaviour in `pcx/core/_tree.py`.
"""

import jax.numpy as jnp
import pytest
from conftest import assert_allclose, count_leaves

import pcx
from pcx.core._tree import _BaseParamRef, tree_apply, tree_extract, tree_inject, tree_ref, tree_unref


class Leafy(pcx.Module):
    def __init__(self, n: int = 2):
        super().__init__()
        self.w = pcx.Param(jnp.ones(n))
        self.b = pcx.Param(jnp.zeros(n))


def test_ref_unref_restores_object_identity_for_an_aliased_param():
    """The key property. A parameter referenced twice must come back out of the
    round-trip as one object referenced twice, not as two equal objects."""
    p = pcx.Param(jnp.ones(2))
    other = pcx.Param(jnp.zeros(2))

    out = tree_unref(tree_ref([p, other, p]))

    assert out[2] is out[0]


def test_ref_unref_returns_the_original_param_objects():
    """Stronger than internal consistency: the caller still holds `p`, and a
    transform writes its result back through these references, so the restored
    tree must point at the caller's object rather than an equivalent copy."""
    p = pcx.Param(jnp.ones(2))

    out = tree_unref(tree_ref([p, p]))

    assert out[0] is p
    assert out[1] is p


@pytest.mark.parametrize("repeats", [1, 2, 5])
def test_ref_contracts_the_leaf_count_to_the_unique_params(repeats):
    """Leaf count is what a jax transform sizes its work by. `n` references to
    one parameter must present as one leaf, or the transform would allocate `n`
    gradients for a single weight and apply `n` updates per step."""
    p = pcx.Param(jnp.ones(2))
    tree = [p] * repeats

    assert count_leaves(tree) == repeats
    assert count_leaves(tree_ref(tree)) == 1


def test_ref_leaves_a_tree_of_distinct_params_untouched():
    """Reffing must be a no-op when there is nothing to deduplicate; otherwise
    every ordinary model would pay a structural change for a feature it does not
    use."""
    tree = [pcx.Param(jnp.ones(2)), pcx.Param(jnp.zeros(2))]

    reffed = tree_ref(tree)

    assert count_leaves(reffed) == 2
    assert reffed[0] is tree[0]
    assert reffed[1] is tree[1]


def test_a_reference_placeholder_contributes_no_dynamic_leaves():
    """A reference is an index, not data. If it were dynamic, the very
    duplication it removes would come back as an extra leaf."""
    assert count_leaves(_BaseParamRef(0)) == 0


def test_ref_and_unref_agree_on_index_order():
    """Indices are positional: unref rebuilds its table by walking the tree in
    the same order ref numbered it. If the two disagreed, an alias would resolve
    to the *wrong* parameter — weights silently swapped, with no error."""
    a = pcx.Param(jnp.array([1.0]))
    b = pcx.Param(jnp.array([2.0]))

    out = tree_unref(tree_ref([a, b, b, a]))

    assert [x is y for x, y in zip(out, [a, b, b, a], strict=True)] == [True] * 4


def test_ref_unref_roundtrips_when_the_first_occurrence_is_nested():
    """Traversal order, not source order, decides which occurrence is the
    original — so the first occurrence may sit arbitrarily deep."""
    p = pcx.Param(jnp.ones(2))

    out = tree_unref(tree_ref([{"deep": p}, p, (p,)]))

    assert out[1] is out[0]["deep"]
    assert out[2][0] is out[0]["deep"]


def test_nested_ref_unref_roundtrips():
    """Documented in NOTE #2: reffing an already-reffed tree is allowed as long
    as unreffing happens in the reverse order. Nested transforms — `jit(vmap(f))`
    — produce exactly this, one ref per layer of nesting."""
    p = pcx.Param(jnp.ones(2))
    tree = [p, p]

    out = tree_unref(tree_unref(tree_ref(tree_ref(tree))))

    assert out[0] is p
    assert out[1] is p


def test_double_reffing_does_not_reintroduce_leaves():
    """Each extra ref layer must remain leaf-neutral, or nesting transforms would
    grow the dynamic tree with every level."""
    p = pcx.Param(jnp.ones(2))

    assert count_leaves(tree_ref(tree_ref([p, p, p]))) == 1


def test_unref_is_a_no_op_on_a_tree_without_references():
    """Transforms call unref unconditionally; on an ordinary tree it must return
    the same parameters untouched."""
    p = pcx.Param(jnp.ones(2))
    q = pcx.Param(jnp.zeros(2))

    out = tree_unref([p, q])

    assert out[0] is p
    assert out[1] is q


def test_a_param_shared_between_two_modules_roundtrips():
    """The realistic case: two layers tied to one weight matrix. This is what
    `pcx.nn.shared` produces and what has to survive a transform intact."""
    shared_w = pcx.Param(jnp.ones(2))
    first, second = Leafy(), Leafy()
    first.w = shared_w
    second.w = shared_w

    assert count_leaves((first, second)) == 4
    assert count_leaves(tree_ref((first, second))) == 3

    out = tree_unref(tree_ref((first, second)))

    assert out[0].w is out[1].w
    assert out[0].w is shared_w


def test_a_mixed_tree_of_params_statics_and_aliases_roundtrips():
    """Statics are `BaseParam`s too, so they take part in the same deduplication.
    A model mixes all three kinds in one tree, and the round-trip must not lose
    the static payload while restoring the dynamic aliases."""
    p = pcx.Param(jnp.ones(2))
    s = pcx.static("activation")
    tree = {"a": p, "b": s, "c": [p, s]}

    reffed = tree_ref(tree)
    out = tree_unref(reffed)

    assert count_leaves(reffed) == 1
    assert out["c"][0] is p
    assert out["c"][1] is s
    assert out["c"][1].get() == "activation"


def test_ref_does_not_disturb_the_values():
    """Reffing is purely structural: it must never touch what a parameter holds."""
    p = pcx.Param(jnp.array([1.0, 2.0]))

    out = tree_unref(tree_ref([p, p]))

    assert_allclose(out[0].get(), jnp.array([1.0, 2.0]))


def test_extract_yields_one_entry_per_dynamic_param_in_traversal_order():
    """`extract`/`inject` are a positional protocol — the two must agree on the
    sequence, so extraction has to be deterministic and skip statics.

    A module flattens by sorted attribute name, deliberately, so that its structure
    does not depend on the order a constructor happened to assign in. `Leafy` assigns
    `w` then `b`, so `b` is extracted first.
    """
    m = Leafy()
    m.w.set(jnp.array([1.0, 2.0]))
    m.b.set(jnp.array([3.0, 4.0]))

    extracted = tree_extract(m, is_pytree=True)

    assert len(extracted) == 2
    assert_allclose(extracted[0].get(), jnp.array([3.0, 4.0]))
    assert_allclose(extracted[1].get(), jnp.array([1.0, 2.0]))


def test_extract_then_inject_transfers_values_in_order():
    """The round-trip: values pulled out of one module and pushed into a
    structurally identical one must land on the matching parameters. This is how
    a transform returns its results to the caller's model."""
    src, dst = Leafy(), Leafy()
    src.w.set(jnp.array([1.0, 2.0]))
    src.b.set(jnp.array([3.0, 4.0]))

    tree_inject(dst, values=tree_extract(src, is_pytree=True), is_pytree=True)

    assert_allclose(dst.w.get(), jnp.array([1.0, 2.0]))
    assert_allclose(dst.b.get(), jnp.array([3.0, 4.0]))


def test_inject_writes_into_the_existing_param_objects():
    """Injection must mutate rather than replace, or the caller's module would
    keep the old parameters and the results would be dropped."""
    src, dst = Leafy(), Leafy()
    src.w.set(jnp.array([5.0, 6.0]))
    target = dst.w

    tree_inject(dst, values=tree_extract(src, is_pytree=True), is_pytree=True)

    assert dst.w is target
    assert_allclose(target.get(), jnp.array([5.0, 6.0]))


def test_inject_from_a_params_tree_matches_injecting_from_values():
    """The `params=` path is the same protocol with the source given as a tree;
    both spellings must produce the same assignment."""
    src, dst = Leafy(), Leafy()
    src.w.set(jnp.array([7.0, 8.0]))

    tree_inject(dst, params=src, is_pytree=True)

    assert_allclose(dst.w.get(), jnp.array([7.0, 8.0]))


def test_inject_leaves_static_params_alone():
    """Statics are excluded by the default filter; consuming a value for one
    would shift every subsequent value onto the wrong parameter."""
    src, dst = Leafy(), Leafy()
    src.w.set(jnp.array([1.0, 2.0]))
    dst.tag = pcx.static("keep")

    tree_inject(dst, values=tree_extract(src, is_pytree=True), is_pytree=True)

    assert dst.tag.get() == "keep"


def test_inject_strict_rejects_surplus_values():
    """`strict=True` is the guard against a caller injecting into a differently
    shaped tree — the failure mode it catches is values landing on the wrong
    parameters, which is otherwise silent."""
    src, dst = Leafy(), Leafy()
    surplus = (*tree_extract(src, is_pytree=True), pcx.Param(jnp.ones(2)))

    with pytest.raises(ValueError, match="number of values"):
        tree_inject(dst, values=surplus, is_pytree=True)


@pytest.mark.bug(
    "BUGS.md#14: too few values raises a bare StopIteration from next() instead of the documented count mismatch"
)
def test_inject_strict_rejects_missing_values():
    """Documented: with `strict=True` "the number of values must match the number
    of leaves in the pytree" — a mismatch in either direction is the same user
    error and deserves the same `ValueError`.

    A bare `StopIteration` is actively harmful: raised inside a generator it is
    converted by PEP 479 into an unrelated `RuntimeError`, and raised inside any
    enclosing iteration it can terminate the loop silently instead of reporting
    the mismatch. Worse, the parameters visited before the values ran out have
    already been overwritten.
    """
    src, dst = Leafy(), Leafy()
    too_few = tree_extract(src, is_pytree=True)[:1]

    with pytest.raises(ValueError, match="number of values"):
        tree_inject(dst, values=too_few, is_pytree=True)


@pytest.mark.bug(
    "BUGS.md#13: tree_inject calls .get() on every element of `values`, so plain arrays raise AttributeError"
)
def test_inject_accepts_plain_values():
    """`values` is documented as an "input sequence of values to inject", and the
    default `inject_fn` is `lambda n, v: n.set(v)` — a plain value is exactly what
    it expects to receive.

    `tree_inject` instead calls `.get()` on each element, so `values` must in fact
    be a sequence of parameters. That makes the documented pairing with a custom
    `extract_fn` unusable: extracting anything other than the parameter itself —
    a scaled gradient, a masked copy, an array straight out of a transform —
    fails on injection.
    """
    dst = Leafy()

    tree_inject(dst, values=[jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])], is_pytree=True)

    assert_allclose(dst.w.get(), jnp.array([1.0, 2.0]))
    assert_allclose(dst.b.get(), jnp.array([3.0, 4.0]))


def test_tree_apply_visits_every_matching_node():
    """`tree_apply` is how `clear_params` and the mode setters reach into a model;
    a node it skipped would keep stale state through the next step."""
    m = Leafy()

    tree_apply(lambda p: p.set(p.get() + 1.0), lambda x: isinstance(x, pcx.Param), m)

    assert_allclose(m.w.get(), jnp.full((2,), 2.0))
    assert_allclose(m.b.get(), jnp.ones(2))


def test_tree_apply_visits_an_aliased_param_once_per_occurrence():
    """Documented behaviour, with a worked example in the docstring: `tree_apply`
    walks the pydag as written, so a parameter referenced twice is visited twice.

    This is the caller's problem to account for — and the reason `clear_params`
    passes an idempotent function. Pinning it here means a change to the
    traversal cannot pass unnoticed.
    """
    p = pcx.Param(jnp.array(1.0))

    tree_apply(lambda x: x.set(x.get() + 1.0), lambda x: isinstance(x, pcx.Param), [p, p])

    assert_allclose(p.get(), 3.0)


def test_tree_apply_without_recursion_stops_at_the_first_match():
    """Documented: `recursive=False` stops "after the first generation of nodes
    matching filter_fn". A matched node becomes a leaf, so its own children are
    not visited — which is what makes the flag cheap for parameter-level work."""
    visited = []

    class Outer(pcx.Module):
        def __init__(self):
            super().__init__()
            self.inner = Leafy()

    outer = Outer()

    tree_apply(visited.append, lambda x: isinstance(x, pcx.Module), outer, False)

    assert visited == [outer]
