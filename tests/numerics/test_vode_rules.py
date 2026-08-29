"""The Vode ruleset: the state machine that drives every predictive coding network.

A `Ruleset` maps the module's *status* (`init`, a training phase, `None`) to a set of
rewrite rules. Input rules `target <- key:transformation` fire on `Vode.set`; output
rules `key -> target:transformation` fire on `Vode.get`. This is the library's
control flow: which rule fires decides whether a node is forward-initialised, whether
a prediction is clamped, and what the energy is computed from.

Every expectation here comes from the docstrings of `Ruleset`, `Vode.set` and
`Vode.get` in `pcx/predictive_coding/_vode.py` — they specify the syntax, the
transformation chaining order, the "all input rules, first output rule only" policy
and the `get` memoisation — plus the arithmetic of the transformations the tests
themselves define. Nothing is read back from the implementation.

The failure mode this file is aimed at is the quiet one. A rule that fires when it
should not, or that fires on the wrong key, does not raise: it writes a real array
into a real parameter, and the network keeps training on a state machine that is not
the one the user described.
"""

import jax.numpy as jnp
import pytest
from conftest import assert_allclose

import pcx.predictive_coding as pxc

H = jnp.array([[1.0, 2.0], [3.0, 4.0]])
U = jnp.array([[0.5, -0.5], [1.5, -1.5]])


def double(node, key, value, rkg):
    """A transformation, in the signature the Ruleset documents."""
    return value * 2.0


def add_one(node, key, value, rkg):
    return value + 1.0


# Input rules (fired by `set`) #########################################################


def test_a_custom_transformation_is_applied_to_the_value_being_set():
    """`h <- u:double` means "when `u` is set, store `double(u)` into `h`".

    User-supplied transformations are the extension point of the whole ruleset — a
    clamped node, a decayed prediction, a noise injection are all written this way.
    If the transformation were skipped, the raw value would be stored and the node
    would quietly behave like a default Vode.
    """
    vode = pxc.Vode(ruleset={pxc.STATUS.ALL: ("h, u <- u:double",)}, tforms={"double": double})

    vode.set("u", U)

    assert_allclose(vode.h.get(), 2.0 * U)
    assert_allclose(vode.get("u"), 2.0 * U, err_msg="both comma-separated targets receive the transformed value")


def test_chained_transformations_are_applied_left_to_right():
    """`Vode.set` documents chaining as `h <- u:se:zero`, read left to right: the
    leftmost transformation sees the incoming value and each later one sees the
    previous result.

    The transformations here do not commute — `2 * (u + 1)` is not `2 * u + 1` — so
    an inverted chain gives a wrong-but-plausible number rather than an error.
    """
    vode = pxc.Vode(
        ruleset={pxc.STATUS.ALL: ("h <- u:add_one:double",)},
        tforms={"add_one": add_one, "double": double},
    )

    vode.set("u", U)

    assert_allclose(vode.h.get(), 2.0 * (U + 1.0))


def test_every_matching_input_rule_fires_and_the_last_write_to_a_target_wins():
    """`Ruleset` documents that "if multiple input rules match ... they are all
    executed in the order they are specified".

    Both halves matter. All of them firing is how one incoming activation feeds
    several parameters (`h`, an error term, a running statistic). The order is what
    makes a later rule able to refine an earlier one; if the sequence were reversed
    or short-circuited, the target would hold the wrong stage of the pipeline.
    """
    tforms = {"add_one": add_one, "double": double}

    fan_out = pxc.Vode(ruleset={pxc.STATUS.ALL: ("a <- u:add_one", "b <- u:double")}, tforms=tforms)
    fan_out.set("u", U)

    assert_allclose(fan_out.get("a"), U + 1.0, err_msg="the first matching rule did not fire")
    assert_allclose(fan_out.get("b"), 2.0 * U, err_msg="the second matching rule did not fire")

    ordered = pxc.Vode(ruleset={pxc.STATUS.ALL: ("h <- u:add_one", "h <- u:double")}, tforms=tforms)
    ordered.set("u", U)

    assert_allclose(ordered.h.get(), 2.0 * U, err_msg="rules did not run in the order they were specified")

    reversed_ = pxc.Vode(ruleset={pxc.STATUS.ALL: ("h <- u:double", "h <- u:add_one")}, tforms=tforms)
    reversed_.set("u", U)

    assert_allclose(reversed_.h.get(), U + 1.0, err_msg="rules did not run in the order they were specified")


def test_setting_one_key_does_not_fire_a_rule_written_for_a_different_key():
    """A rule names the key it reacts to, and `u2` is not `u`.

    `Vode.set` builds its pattern as `f"(.*(?<!\\s))\\s*<-\\s*({key}.*)"`, so the key
    is a *prefix* match: setting `u` fires every rule whose right-hand side starts
    with `u`. A Vode with two incoming activations — `u` from the layer above and
    `u2` from a skip connection, which is exactly what a multi-input node looks like
    — therefore has `u2`'s rule executed with `u`'s value.

    The damage is doubled: because a rule matched, the `len(rules) == 0` fallback
    never runs, so `u` is not stored under its own name at all. The node ends up
    holding one activation in the wrong slot and nothing in the right one.
    """
    vode = pxc.Vode(ruleset={pxc.STATUS.ALL: ("h <- u2",)})
    vode.h.set(H)

    vode.set("u", U)

    assert_allclose(vode.h.get(), H, err_msg="a rule written for 'u2' fired when 'u' was set")
    assert vode.get("u") is not None, "'u' was not stored under its own name"
    assert_allclose(vode.get("u"), U)


# Status matching ######################################################################


def test_status_init_fires_the_forward_initialisation_rule():
    """The default ruleset is `{STATUS.INIT: ("h, u <- u",)}`: during the init pass
    the node's value is seeded with its feed-forward prediction. This is the one rule
    every pcx network relies on, so it anchors the status tests below."""
    vode = pxc.Vode()
    vode.h.set(H)
    vode.status = pxc.STATUS.INIT

    vode.set("u", U)

    assert_allclose(vode.h.get(), U)


def test_status_all_matches_every_status_including_none():
    """`STATUS.ALL` is `".*"`, documented as the pattern that "would apply the two
    rules to any status" — including the `None` status a freshly built model carries,
    which `Ruleset.filter` normalises to the empty string.

    Rules registered under `.*` are the ones that must hold in every phase (a
    clamping rule on an input node, say). One that silently stopped firing outside a
    named phase would leave the node unclamped for the whole of inference.
    """
    for status in (pxc.STATUS.NONE, pxc.STATUS.INIT, "train", "eval"):
        vode = pxc.Vode(ruleset={pxc.STATUS.ALL: ("a <- u:double",)}, tforms={"double": double})
        vode.status = status

        vode.set("u", U)

        assert_allclose(vode.get("a"), 2.0 * U, err_msg=f"a '.*' rule did not fire under status {status!r}")


def test_status_none_does_not_fire_a_rule_registered_for_a_named_status():
    """`STATUS.NONE` means no phase is active, so a rule registered for `init` must
    not fire.

    This is the state a model is in during ordinary inference and evaluation. If the
    init rule fired here, `h` would be overwritten by the incoming prediction on
    every forward pass, the prediction error would collapse to zero, and inference
    would appear to converge instantly while learning nothing.
    """
    vode = pxc.Vode(ruleset={pxc.STATUS.ALL: ("a <- u",)})
    vode.h.set(H)
    vode.status = pxc.STATUS.NONE

    vode.set("u", U)

    assert_allclose(vode.h.get(), H, err_msg="the 'init' rule fired under STATUS.NONE")
    assert_allclose(vode.get("a"), U, err_msg="the '.*' rule did not fire under STATUS.NONE")


@pytest.mark.parametrize("status", ["initialise", "initialisation", "init_weights", "initial"], ids=lambda s: s)
def test_a_status_is_matched_exactly_and_not_by_prefix(status: str):
    """A status must match a rule's pattern as a whole, not merely start with it.

    `Ruleset.filter` calls `re.match(pattern, status)`, which anchors only at the
    start, so the pattern `"init"` behaves as if it were written `"init.*"`. The
    class docstring documents the patterns as regular expressions and offers `.*` as
    *the* way to say "any status" — which is meaningless if every pattern already
    carries an implicit one.

    The consequence is silent and specific: a user who names a phase `"initialise"`,
    `"init_weights"` or `"initial"` gets the built-in `STATUS.INIT` rule
    `h, u <- u` executed underneath their own, so `h` is overwritten with the
    incoming prediction and every energy computed in that phase is zero.
    """
    vode = pxc.Vode()
    vode.h.set(H)
    vode.status = status

    vode.set("u", U)

    assert_allclose(vode.h.get(), H, err_msg=f"status {status!r} fired the rule registered for 'init'")


# Output rules (fired by `get`) ########################################################


def active_vode(rules, tforms):
    """A Vode with the given output rules that has already received an activation.

    Setting `u` is what a node does on every forward pass, so this is the state a
    real network holds one in. It also side-steps a defect that would otherwise mask
    everything in this section: `VodeParam.Cache()` starts life wrapping `None`, and
    `ParamDict.get` does not guard against that (BUGS.md#7), so evaluating an output
    rule on a Vode that has never been written to raises `AttributeError` before the
    rule is reached. `test_an_output_rule_works_on_a_freshly_built_vode` asserts that
    case directly; the tests below are about the rules themselves.
    """
    vode = pxc.Vode(ruleset={pxc.STATUS.ALL: rules}, tforms=tforms)
    vode.h.set(H)
    vode.set("u", U)

    return vode


def test_an_output_rule_applies_its_transformation_to_the_referenced_parameter():
    """`e -> h:double` means "when `e` is asked for, return `double(h)`".

    Output rules are how a Vode exposes a derived quantity — an error signal, a
    normalised activation — under its own name, so that the rest of the network can
    ask for it without knowing how it is built.
    """
    vode = active_vode(("e -> h:double",), {"double": double})

    assert_allclose(vode.get("e"), 2.0 * H)


@pytest.mark.bug(
    "BUGS.md#7: VodeParam.Cache() wraps None until something is written to it, and ParamDict.get does not guard "
    "against that, so evaluating an output rule on a freshly built Vode raises AttributeError"
)
def test_an_output_rule_works_on_a_freshly_built_vode():
    """An output rule must be evaluable the moment the Vode exists.

    `Vode.get` looks the rule's right-hand side up in the cache first, to reuse a
    memoised result. On a new node that lookup is simply a miss — and a miss has an
    answer, `None`, which is what the rest of `apply_get_transformation` is written
    to handle. Instead `ParamDict.get` raises `AttributeError: 'NoneType' object has
    no attribute 'get'`, because the cache wraps `None` rather than `{}` until the
    first write.

    So a custom output rule is unusable until something else has incidentally
    populated the cache. BUGS.md#7 records this for a *cleared* cache; the same
    missing guard makes it the state of every cache at construction.
    """
    vode = pxc.Vode(ruleset={pxc.STATUS.ALL: ("e -> h:double",)}, tforms={"double": double})
    vode.h.set(H)

    assert_allclose(vode.get("e"), 2.0 * H)


def test_a_get_rule_is_computed_once_and_then_served_from_the_cache():
    """Documented in `Vode.get`: "the right-hand side of the rule is also saved to the
    cache, so subsequent calls to the same key will return the same value without
    recomputation".

    Memoisation is not just a speed knob here. A transformation may draw randomness
    or read a mutable cache, so recomputing would return a *different* value on the
    second read, and two parts of the network asking for the same derived quantity
    would disagree about it within one step.
    """
    calls = []

    def counted(node, key, value, rkg):
        calls.append(key)
        return value * 2.0

    vode = active_vode(("e -> h:counted",), {"counted": counted})

    first = vode.get("e")
    second = vode.get("e")

    assert_allclose(first, 2.0 * H)
    assert_allclose(second, 2.0 * H, err_msg="the memoised read returned a different value")
    assert len(calls) == 1, f"the transformation was recomputed: called {len(calls)} times"


def test_a_get_rule_reflects_a_new_value_after_the_cache_is_cleared():
    """The other half of the memoisation contract: `clear_params` runs between steps,
    and after it the derived value must be recomputed from the *current* state.

    A cache that outlived the clear would pin an error signal to the first batch's
    value for the rest of training.
    """
    vode = active_vode(("e -> h:double",), {"double": double})
    assert_allclose(vode.get("e"), 2.0 * H)

    vode.clear_params(pxc.VodeParam.Cache)
    vode.h.set(H + 1.0)
    # The next forward pass, which also repopulates the cleared cache dict.
    vode.set("u", U)

    assert_allclose(vode.get("e"), 2.0 * (H + 1.0))


@pytest.mark.filterwarnings("ignore:Multiple output rules matched")
def test_only_the_first_matching_output_rule_is_applied():
    """`Ruleset` documents that "if multiple output rules match the current status and
    operation, only the first one is executed".

    A `get` returns one value, so the policy has to be deterministic and it has to be
    the one the user can predict from the order they wrote their rules in. Picking
    the last, or the first in dictionary order, would make the returned quantity
    depend on something the user never specified.
    """
    vode = active_vode(("e -> h:double", "e -> h:add_one"), {"double": double, "add_one": add_one})

    assert_allclose(vode.get("e"), 2.0 * H)


def test_multiple_matching_output_rules_report_through_the_warnings_module():
    """An ambiguous ruleset — two output rules matching one key — has to be reported
    as a real Python warning.

    A bare `print` cannot be filtered, captured, routed to a log, or promoted to an
    error with `-W error`, so a CI run cannot fail on it and a notebook user scrolls
    past it. Worse, it would be emitted while a `jit`ted step is being *traced*: it
    would appear once, at compile time, detached from the step it belongs to, and
    never again for the thousands of steps that follow.

    Meanwhile the effect it warns about is silent data loss — one of the two rules is
    discarded — so this is precisely the situation that warrants a `UserWarning` the
    user can turn into an exception.
    """
    vode = active_vode(("e -> h:double", "e -> h:add_one"), {"double": double, "add_one": add_one})

    with pytest.warns(UserWarning, match="Multiple output rules"):
        vode.get("e")


# `__call__` plumbing ##################################################################


def test_call_stores_the_extra_activations_passed_as_keyword_arguments():
    """`Vode.__call__` documents `**kwargs` as "optional additional activations to
    set", each routed through the ruleset exactly as `u` is.

    This is how a node receives a second input — a target to clamp to, a top-down
    prediction — in the same call as its feed-forward one.
    """
    vode = pxc.Vode()
    target = jnp.zeros_like(H)

    vode(U, target=target)

    assert_allclose(vode.get("u"), U)
    assert_allclose(vode.get("target"), target)


def test_call_with_output_none_returns_the_vode_itself():
    """Documented: "If 'None', the Vode object is returned". Returning the node lets
    a caller keep chaining (`vode(u, output=None).get("e")`) instead of holding a
    separate reference."""
    vode = pxc.Vode()

    assert vode(U, output=None) is vode


def test_call_returns_the_parameter_named_by_output():
    """`output` selects which value comes back, so a node can hand its caller the
    incoming activation or a derived quantity rather than its state."""
    vode = pxc.Vode(ruleset={pxc.STATUS.ALL: ("e -> h:double",)}, tforms={"double": double})
    vode.h.set(H)

    assert_allclose(vode(U, output="u"), U)
    assert_allclose(vode(U, output="e"), 2.0 * H)


# Energy #############################################################################


def test_a_vode_without_an_energy_function_contributes_zero_energy():
    """A node built with `energy_fn=None` is unconstrained: it holds state but places
    no term in the objective, so its energy is the additive identity.

    Returning anything else — or raising — would corrupt the sum that
    `EnergyModule.energy` reduces over, which *is* the variational free energy being
    minimised.
    """
    vode = pxc.Vode(energy_fn=None)
    vode.h.set(H)
    vode.set("u", U)

    assert_allclose(vode.energy(), 0.0)
