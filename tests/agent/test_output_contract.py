"""Tests for the Axis-3 structured-output contract (pre-registration section 5.4).

Two things are under test here and they fail for different reasons.

THE ATTACHMENT RULE (`agent.WfedsAgent`): `response_format` must ride on the
POST-TOOL completion call only, and only in the "structured" arm. Everything the
free arm sends must stay identical to the frozen 2026-09-01 configuration,
because the stored 195 free-text runs are the comparison baseline and nothing may
retroactively move Stage A (the emitted tool arguments) or Stage B (one tool call
per turn). The most load-bearing tests in this file are therefore the ones that
assert what the FIRST call looks like, not the second.

THE CONTRACT ITSELF (`final_answer_schema`): the schema must be legal under
OpenAI strict mode, must accept the ideal answer for a realistic compact payload,
must reject the mutations a sloppy model actually produces, and must not drift
away from the published artefact `scripts/validation/wfeds_final_answer_schema.json`,
which is what the write-up cites.

Everything runs offline: no API key, no network. `agent.get_model` is patched
exactly as `tests/agent/test_agent.py` does it, and litellm is intercepted through
`sys.modules` (agent.py imports it lazily inside `chat()`).
"""

import copy
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import jsonschema
import pytest

import agent
from agent import WfedsAgent, _response_meta
from final_answer_schema import (FINAL_ANSWER_RESPONSE_FORMAT, FINAL_ANSWER_SCHEMA,
                                 FINAL_ANSWER_SCHEMA_SHA256)
from tests.helpers.fake_litellm import make_response

PUBLISHED_SCHEMA_PATH = (Path(__file__).resolve().parents[2] / "scripts" /
                         "validation" / "wfeds_final_answer_schema.json")

# The frozen fingerprints of the 2026-09-01 configuration. Both arms must send
# the same prompt and the same tool schema, so these two prefixes are the
# tripwire for the whole comparison: if either moves, the stored baseline stops
# being a baseline and the study is void.
FROZEN_SYSTEM_PROMPT_SHA12 = "f005ab5e76d6"
FROZEN_TOOLS_SCHEMA_SHA12 = "0c5ae805199e"
REGISTERED_SCHEMA_SHA256 = (
    "d079a01b22d4c344a4f2b7a6a62b147215860deeeddd373e5b597f8ed498ccf0")


# --------------------------------------------------------------------------------
# Local test helpers
#
# WHY NOT tests/helpers/fake_litellm.install_fake_litellm: that helper's Mock
# records `call_args_list`, but `chat()` passes `messages=self.messages`, i.e. the
# SAME list object on every call, and then keeps appending to it. By the time a
# test inspects call 1 it is looking at the final state of the conversation, so
# per-call assertions about what round 1 contained are impossible through it. The
# recorder below snapshots each call at call time, and reduces every message to a
# (role, content, has_tool_calls) tuple because the fake tool-call objects are
# fresh instances per run and would never compare equal across two agents.
# --------------------------------------------------------------------------------
def _message_shape(msg):
    """The comparable identity of one message: role, content, and whether it
    carried tool calls. Never the message object itself."""
    return (msg.get("role"), msg.get("content"), bool(msg.get("tool_calls")))


class _CallRecord:
    """One `litellm.completion(...)` call, frozen at the moment it was made."""

    def __init__(self, kwargs):
        # `messages` is split out and shaped; everything else is kept verbatim,
        # so `kwargs` is exactly the non-message surface the pre-registration
        # compares between arms.
        self.kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
        self.messages = [_message_shape(m) for m in kwargs.get("messages", [])]

    @property
    def has_response_format(self):
        return "response_format" in self.kwargs


def _install_recording_litellm(monkeypatch, responses):
    """Fake `litellm` whose `.completion` snapshots each call and then returns
    the next scripted response. Returns the list of `_CallRecord`s, which grows
    as the agent runs. Calling more times than the test scripted is a failure,
    not a silent StopIteration, so a runaway loop is reported as what it is."""
    records = []
    remaining = list(responses)

    def _completion(**kwargs):
        records.append(_CallRecord(kwargs))
        if not remaining:
            raise AssertionError(
                "litellm.completion was called more times than the test scripted "
                f"(call {len(records)})")
        return remaining.pop(0)

    fake_module = types.ModuleType("litellm")
    fake_module.completion = _completion
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return records


def _tool_call_response(call_id="call_1"):
    return make_response(tool_calls=[(call_id, "run_fire_scenario",
                                      '{"use_pins": true, "window_km": 12, '
                                      '"horizon_h": 6}')])


@pytest.fixture
def make_agent(monkeypatch):
    """Factory for agents with `get_model()` bypassed, same reason as
    `tests/agent/test_agent.py`'s `wfeds_agent`: llm_config.get_model raises
    unless a real provider key is in the environment. `get_params` is left real,
    so the sampling/provider kwargs under test are the production ones."""
    monkeypatch.setattr(agent, "get_model",
                        lambda preset=None: ("fake/model", "fake-key"))

    def _make(output_contract="free"):
        return WfedsAgent(output_contract=output_contract)

    return _make


@pytest.fixture
def stub_run_tool(monkeypatch):
    """The tool round returns instantly. `_run_tool` is patched on the CLASS as a
    plain Mock, so (as documented in test_agent.py) it is called without `self`."""
    mock = Mock(return_value=('{"run_id": "20260921_000000_test"}', [], []))
    monkeypatch.setattr(WfedsAgent, "_run_tool", mock)
    return mock


# --------------------------------------------------------------------------------
# Where the contract is attached
# --------------------------------------------------------------------------------
class TestResponseFormatAttachment:
    def test_free_contract_never_attaches_response_format(self, make_agent,
                                                          monkeypatch, stub_run_tool):
        """The control condition. Both rounds of a full tool-using turn must go
        out exactly as they did before the contract existed."""
        calls = _install_recording_litellm(monkeypatch, [
            _tool_call_response(), make_response(content="ανάλυση")])
        wfeds_agent = make_agent("free")

        wfeds_agent.chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])

        assert len(calls) == 2
        assert not calls[0].has_response_format
        assert not calls[1].has_response_format

    def test_structured_attaches_contract_only_on_the_post_tool_call(
            self, make_agent, monkeypatch, stub_run_tool):
        """The treatment, and the single line that defines it: round 1 is the
        call that produces the tool arguments (Stage A) and must stay untouched;
        round 2 is the call that produces the narrative (Stage C) and is the only
        one the contract is allowed to reach."""
        calls = _install_recording_litellm(monkeypatch, [
            _tool_call_response(), make_response(content='{"status": "ok"}')])
        wfeds_agent = make_agent("structured")

        wfeds_agent.chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])

        assert len(calls) == 2
        assert not calls[0].has_response_format
        assert calls[1].kwargs["response_format"] is FINAL_ANSWER_RESPONSE_FORMAT

    def test_structured_second_turn_does_not_attach_on_its_first_call(
            self, make_agent, monkeypatch, stub_run_tool):
        """THE regression this whole design turns on. The conversation persists
        across turns (telegram_bot.py and docker/api.py both keep one agent per
        chat), so turn 2 starts with a stale role:"tool" message from turn 1 in
        self.messages. If the attachment condition were read off the message
        history instead of a turn-local flag, turn 2's FIRST call would carry the
        contract, and that call is the one that emits the tool arguments: Stage A
        and Stage B would silently move and the comparison would be worthless."""
        calls = _install_recording_litellm(monkeypatch, [
            _tool_call_response("call_1"), make_response(content='{"status": "ok"}'),
            _tool_call_response("call_2"), make_response(content='{"status": "ok"}')])
        wfeds_agent = make_agent("structured")

        wfeds_agent.chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])
        wfeds_agent.chat("και για 12 ώρες;", pins=[(38.9, 23.1)])

        assert len(calls) == 4
        turn_two_first_call = calls[2]
        # The precondition of the regression: the stale tool message really is
        # still in the history the second turn sends.
        assert any(role == "tool" for role, _c, _t in turn_two_first_call.messages)
        assert not turn_two_first_call.has_response_format
        # ...and the rule still applies normally later in the same turn.
        assert calls[3].kwargs["response_format"] is FINAL_ANSWER_RESPONSE_FORMAT

    @pytest.mark.parametrize("contract", ["free", "structured"])
    def test_tools_are_passed_on_every_call_in_both_modes(
            self, make_agent, monkeypatch, stub_run_tool, contract):
        """`tools=TOOLS` is unconditional. A contract that suppressed the tool
        schema on the post-tool round would change what the model can still do
        and would not be a pure output-format intervention."""
        calls = _install_recording_litellm(monkeypatch, [
            _tool_call_response(), make_response(content="τέλος")])
        wfeds_agent = make_agent(contract)

        wfeds_agent.chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])

        assert len(calls) == 2
        for rec in calls:
            assert rec.kwargs["tools"] is agent.TOOLS


# --------------------------------------------------------------------------------
# Selecting the contract
# --------------------------------------------------------------------------------
class TestContractSelection:
    def test_unknown_output_contract_raises_value_error(self, make_agent):
        """Fail at construction, not silently at the first completion call: a
        typo in the harness would otherwise produce a whole arm of free runs
        labelled "structured"."""
        with pytest.raises(ValueError) as excinfo:
            make_agent("bogus")

        assert "bogus" in str(excinfo.value)

    def test_default_output_contract_is_free(self, make_agent):
        """Every existing caller (telegram_bot, docker/api, main) constructs the
        agent without the new argument, so the default IS the frozen behaviour."""
        assert make_agent().output_contract == "free"

    def test_the_only_contracts_are_free_and_structured(self):
        assert agent.OUTPUT_CONTRACTS == ("free", "structured")


# --------------------------------------------------------------------------------
# The frozen configuration: no per-arm prompt, ever
# --------------------------------------------------------------------------------
class TestFrozenConfiguration:
    def test_prompt_and_tool_schema_hashes_are_the_frozen_ones(self):
        """These two prefixes pin the 2026-09-01 configuration that produced the
        stored baseline. Any edit to SYSTEM_PROMPT or TOOLS, however cosmetic,
        breaks the comparability of the two arms and must fail here first."""
        assert agent.SYSTEM_PROMPT_SHA256[:12] == FROZEN_SYSTEM_PROMPT_SHA12
        assert agent.TOOLS_SCHEMA_SHA256[:12] == FROZEN_TOOLS_SCHEMA_SHA12

    def test_no_per_arm_prompt_constant_exists_in_the_module(self):
        """Option A (a second, structured-only system prompt) was rejected: it
        would have forked the prompt and contaminated Stage A and Stage B. This
        asserts on the MODULE NAMESPACE rather than on a hash, so the rejected
        design cannot reappear under any name we can foresee."""
        for forbidden in ("SYSTEM_PROMPT_STRUCTURED", "PROMPTS",
                          "SYSTEM_PROMPTS", "STRUCTURED_SYSTEM_PROMPT"):
            assert not hasattr(agent, forbidden), (
                f"agent.{forbidden} exists: the per-arm prompt is back and the "
                f"treatment is no longer output-format only")

    def test_both_arms_send_the_very_same_prompt_object(self, make_agent):
        """Object identity, not string equality: a copy, a reformat or an
        f-string rebuild would each pass an == check and still be a second,
        independently editable prompt."""
        free_prompt = make_agent("free").messages[0]["content"]
        structured_prompt = make_agent("structured").messages[0]["content"]

        assert free_prompt is agent.SYSTEM_PROMPT
        assert structured_prompt is agent.SYSTEM_PROMPT


# --------------------------------------------------------------------------------
# The free path must behave exactly as the pre-contract agent did
# --------------------------------------------------------------------------------
class TestFreePathUnchanged:
    def test_free_mode_sends_exactly_the_pre_contract_kwargs(
            self, make_agent, monkeypatch, stub_run_tool):
        """Before the contract existed the call was
        `completion(model=, api_key=, messages=, tools=, **self.params)`. The key
        SET is asserted, not just the absence of response_format, so a future
        stray kwarg on the free path is caught too."""
        calls = _install_recording_litellm(monkeypatch, [
            _tool_call_response(), make_response(content="ανάλυση")])
        wfeds_agent = make_agent("free")

        wfeds_agent.chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])

        expected_keys = {"model", "api_key", "tools"} | set(wfeds_agent.params)
        for rec in calls:
            assert set(rec.kwargs) == expected_keys
            assert rec.kwargs["model"] == "fake/model"
            assert rec.kwargs["api_key"] == "fake-key"
            assert rec.kwargs["tools"] is agent.TOOLS
            for name, value in wfeds_agent.params.items():
                assert rec.kwargs[name] == value
        # The conversation shape is the old one as well: system + user on round 1,
        # plus the assistant tool-call message and the tool result on round 2.
        assert calls[0].messages == [
            ("system", agent.SYSTEM_PROMPT, False),
            ("user", "[ΤΡΕΧΟΝΤΑ PINS ΧΡΗΣΤΗ: [(38.9, 23.1)]]\nτρέξε 6 ώρες", False)]
        assert [role for role, _c, _t in calls[1].messages] == [
            "system", "user", "assistant", "tool"]

    def test_round_one_is_identical_across_the_two_arms(
            self, make_agent, monkeypatch, stub_run_tool):
        """The executable form of the Stage-A and Stage-B immunity claim: drive
        one free agent and one structured agent through the same turn and compare
        their FIRST call. Same model, same params, same tools, same two messages,
        and no response_format on either. If this passes, the only channel left
        through which the treatment could move Stage A is a second tool call on a
        later round, which is the pre-declared validity condition checked after
        the runs, not here."""
        free_calls = _install_recording_litellm(monkeypatch, [
            _tool_call_response(), make_response(content="ανάλυση")])
        make_agent("free").chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])

        structured_calls = _install_recording_litellm(monkeypatch, [
            _tool_call_response(), make_response(content='{"status": "ok"}')])
        make_agent("structured").chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])

        assert free_calls[0].kwargs == structured_calls[0].kwargs
        assert free_calls[0].messages == structured_calls[0].messages
        assert not free_calls[0].has_response_format
        assert not structured_calls[0].has_response_format


# --------------------------------------------------------------------------------
# The schema artefact: one source of truth
# --------------------------------------------------------------------------------
class TestSchemaArtefact:
    def test_schema_sha256_is_the_registered_fingerprint(self):
        """The hash written into every structured run's provenance.json and
        quoted in the pre-registration. It is what makes a stored run traceable
        to the exact contract that produced it."""
        assert FINAL_ANSWER_SCHEMA_SHA256 == REGISTERED_SCHEMA_SHA256

    def test_module_schema_and_published_artefact_cannot_drift(self):
        """The write-up cites the JSON file; the runs use the Python object. Two
        copies of anything drift eventually, so they are compared here in the
        same canonical serialisation that is hashed."""
        published = json.loads(PUBLISHED_SCHEMA_PATH.read_text(encoding="utf-8"))

        def _canonical(obj):
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)

        assert _canonical(published["json_schema"]["schema"]) == _canonical(
            FINAL_ANSWER_SCHEMA), "the inner schema drifted"
        # The wrapper matters too: a published copy carrying strict=False
        # would describe a run the provider never enforced.
        assert _canonical(published) == _canonical(
            FINAL_ANSWER_RESPONSE_FORMAT), "the response_format wrapper drifted"

    def test_response_format_wrapper_is_a_strict_named_json_schema(self):
        """Strict mode is the whole point: without `strict: True` the provider
        treats the schema as a hint and the validity numbers mean nothing."""
        assert FINAL_ANSWER_RESPONSE_FORMAT["type"] == "json_schema"
        wrapper = FINAL_ANSWER_RESPONSE_FORMAT["json_schema"]
        assert wrapper["name"] == "wfeds_final_answer"
        assert wrapper["strict"] is True
        assert wrapper["schema"] is FINAL_ANSWER_SCHEMA


# --------------------------------------------------------------------------------
# Strict-mode legality, asserted structurally rather than by calling a provider
# --------------------------------------------------------------------------------
# The subset OpenAI strict mode accepts is narrow, and a schema it rejects fails
# at request time on all 195 runs, i.e. after the money is spent. These keywords
# are the ones it does NOT support; walking the tree for them offline is the only
# way to find that out before the pilot.
UNSUPPORTED_KEYWORDS = ("minItems", "maxItems", "minLength", "maxLength", "pattern",
                        "format", "minimum", "maximum", "exclusiveMinimum",
                        "exclusiveMaximum", "multipleOf", "uniqueItems",
                        "minProperties", "maxProperties", "anyOf", "oneOf", "allOf",
                        "not", "if", "then", "else", "propertyNames",
                        "patternProperties", "$ref", "const", "default", "nullable")


def _walk_schema(node, path="$"):
    """Yield (path, node) for every dict node in the schema tree."""
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from _walk_schema(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_schema(value, f"{path}[{i}]")


def _object_nodes():
    return [(path, node) for path, node in _walk_schema(FINAL_ANSWER_SCHEMA)
            if node.get("type") == "object"]


class TestStrictModeLegality:
    def test_root_is_an_object(self):
        assert FINAL_ANSWER_SCHEMA["type"] == "object"

    def test_every_object_node_forbids_additional_properties(self):
        nodes = _object_nodes()
        assert len(nodes) == 4        # the root plus the three array item objects
        for path, node in nodes:
            assert node.get("additionalProperties") is False, path

    def test_every_required_list_is_the_full_property_list(self):
        """Strict mode has no optional fields: every property must be required.
        A field the model is allowed to omit is a field the scorer cannot count."""
        for path, node in _object_nodes():
            required = node.get("required", [])
            properties = node.get("properties", {})
            assert len(required) == len(set(required)), path
            assert set(required) == set(properties), path

    def test_no_unsupported_keyword_appears_anywhere(self):
        for path, node in _walk_schema(FINAL_ANSWER_SCHEMA):
            for keyword in UNSUPPORTED_KEYWORDS:
                assert keyword not in node, f"{path} carries {keyword}"


# --------------------------------------------------------------------------------
# Instance validation: the ideal answer passes, the realistic failures do not
# --------------------------------------------------------------------------------
# A realistic `_compact()` payload: the tool result the model actually reads on
# the post-tool round. cut_off rises at periods 2 and 4 and then holds, so the
# contract's own rules pick out exactly two critical transitions, and final_hour
# is a DIFFERENT hour from the last change_hours row, which is the confusion the
# `final_state` description exists to prevent.
COMPACT_PAYLOAD = {
    "run_id": "20260921_120000_test",
    "inputs": {"mode": "point_ignition", "window_km": 12, "horizon_h": 6,
               "weather_source": "archive"},
    "change_hours": [
        {"period": 0, "fire_km2": 0.4, "at_risk": 0, "population": 0,
         "routed": 0, "cut_off": 0, "impacted": 0},
        {"period": 2, "fire_km2": 1.8, "at_risk": 2, "population": 310,
         "routed": 2, "cut_off": 1, "impacted": 0},
        {"period": 4, "fire_km2": 4.1, "at_risk": 5, "population": 980,
         "routed": 2, "cut_off": 3, "impacted": 1},
        {"period": 5, "fire_km2": 5.0, "at_risk": 6, "population": 1150,
         "routed": 3, "cut_off": 3, "impacted": 1},
    ],
    "final_hour": {"period": 6, "fire_km2": 5.9, "at_risk": 6, "population": 1150,
                   "routed": 3, "cut_off": 3, "impacted": 1, "edges_removed": 11},
}


def _ideal_answer(compact):
    """The answer the contract asks for, built from the payload by the schema's
    own stated rules. Written out rather than hardcoded so the fixture documents
    those rules: the first change_hours row, every row where cut_off INCREASED,
    and final_hour (not the last change_hours row) for the final state."""
    rows = compact["change_hours"]
    transitions = [row for prev, row in zip(rows, rows[1:])
                   if row["cut_off"] > prev["cut_off"]]
    final = compact["final_hour"]
    return {
        "status": "ok",
        "initial_state": [{"period": rows[0]["period"],
                           "settlements_without_route": rows[0]["cut_off"]}],
        "critical_transitions": [
            {"period": r["period"], "settlements_without_route": r["cut_off"]}
            for r in transitions],
        "final_state": [{"period": final["period"],
                         "settlements_without_route": final["cut_off"],
                         "road_segments_removed": final["edges_removed"]}],
        "interpretation": "Η κατάσταση επιδεινώνεται μετά τη 2η ώρα.",
        "limitations": ("Το μοντέλο δεν αποτυπώνει κατάσβεση. Η τελική απόφαση "
                        "ανήκει στον επιχειρησιακό υπεύθυνο."),
    }


def _validate(instance):
    jsonschema.Draft202012Validator(FINAL_ANSWER_SCHEMA).validate(instance)


def _mutated(**overrides):
    """A deep copy of the ideal instance with one thing broken at the root."""
    instance = copy.deepcopy(_ideal_answer(COMPACT_PAYLOAD))
    for key, value in overrides.items():
        instance[key] = value
    return instance


class TestInstanceValidation:
    def test_ideal_instance_from_a_realistic_compact_validates(self):
        instance = _ideal_answer(COMPACT_PAYLOAD)

        _validate(instance)

        # Sanity on the fixture itself: if the payload stopped exercising the
        # transition rule, the mutation tests below would be testing nothing.
        assert [t["period"] for t in instance["critical_transitions"]] == [2, 4]
        assert instance["final_state"][0]["period"] == 6

    def test_extra_key_at_root_is_rejected(self):
        """The commonest failure: the model adds a chatty "summary" field."""
        instance = _mutated(summary="μια σύνοψη")

        with pytest.raises(jsonschema.ValidationError):
            _validate(instance)

    def test_extra_key_in_a_final_state_entry_is_rejected(self):
        instance = copy.deepcopy(_ideal_answer(COMPACT_PAYLOAD))
        instance["final_state"][0]["population"] = 1150

        with pytest.raises(jsonschema.ValidationError):
            _validate(instance)

    def test_string_where_an_integer_is_required_is_rejected(self):
        """A "3" instead of a 3 would be scored as a wrong number downstream, so
        the contract has to stop it at the boundary."""
        instance = copy.deepcopy(_ideal_answer(COMPACT_PAYLOAD))
        instance["final_state"][0]["settlements_without_route"] = "3"

        with pytest.raises(jsonschema.ValidationError):
            _validate(instance)

    def test_unknown_status_value_is_rejected(self):
        instance = _mutated(status="partial")

        with pytest.raises(jsonschema.ValidationError):
            _validate(instance)

    def test_missing_limitations_is_rejected(self):
        """Rule 4 of the system prompt makes the limits statement mandatory; in
        the structured arm that obligation is enforced by `required`."""
        instance = copy.deepcopy(_ideal_answer(COMPACT_PAYLOAD))
        del instance["limitations"]

        with pytest.raises(jsonschema.ValidationError):
            _validate(instance)

    def test_extra_property_in_a_critical_transition_entry_is_rejected(self):
        instance = copy.deepcopy(_ideal_answer(COMPACT_PAYLOAD))
        instance["critical_transitions"][0]["at_risk"] = 2

        with pytest.raises(jsonschema.ValidationError):
            _validate(instance)

    def test_null_in_an_integer_slot_is_rejected(self):
        """"I do not know" must not arrive as null in a counting field: a null
        would be read as a missing count rather than as an abstention, and the
        abstention channel is `status`."""
        instance = copy.deepcopy(_ideal_answer(COMPACT_PAYLOAD))
        instance["initial_state"][0]["settlements_without_route"] = None

        with pytest.raises(jsonschema.ValidationError):
            _validate(instance)

    def test_non_string_interpretation_is_rejected(self):
        """The model that helpfully returns bullets as a list instead of as a
        string: caught, because the Stage-C scorer reads this field as text."""
        instance = _mutated(interpretation=["πρώτο", "δεύτερο"])

        with pytest.raises(jsonschema.ValidationError):
            _validate(instance)


# --------------------------------------------------------------------------------
# Provider metadata capture
# --------------------------------------------------------------------------------
# The 2026-09-01 baseline recorded only the litellm alias, so what the provider
# actually served behind it is unrecoverable and model drift cannot be ruled out
# for the stored arm. That gap is a declared limitation of this study, and these
# tests are what stop it recurring on the new runs.
class _FakeUsage:
    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self, exclude_none=True):
        return {k: v for k, v in self._fields.items()
                if not exclude_none or v is not None}


class _FakeMetaMessage:
    def __init__(self, refusal=None):
        self.refusal = refusal


class _FakeMetaChoice:
    def __init__(self, finish_reason="stop", refusal=None):
        self.finish_reason = finish_reason
        self.message = _FakeMetaMessage(refusal)


class _FakeMetaResponse:
    """A completion as the providers actually return one, with every optional
    field populated. `extra` sets further attributes per test."""

    def __init__(self, finish_reason="stop", refusal=None, **extra):
        self.model = "gpt-5.6-luna-2026-08-14"
        self.id = "chatcmpl-abc123"
        self.created = 1758460800
        self.system_fingerprint = "fp_0a1b2c3d"
        self.service_tier = "default"
        self.choices = [_FakeMetaChoice(finish_reason, refusal)]
        self.usage = _FakeUsage(prompt_tokens=3989, completion_tokens=708,
                                total_tokens=4697, reasoning_tokens=None)
        for key, value in extra.items():
            setattr(self, key, value)


class TestResponseMeta:
    def test_captures_the_served_identity_and_the_usage_object(self):
        meta = _response_meta(_FakeMetaResponse(), True)

        assert meta["model"] == "gpt-5.6-luna-2026-08-14"
        assert meta["id"] == "chatcmpl-abc123"
        assert meta["created"] == 1758460800
        assert meta["system_fingerprint"] == "fp_0a1b2c3d"
        assert meta["service_tier"] == "default"
        assert meta["finish_reason"] == "stop"
        # exclude_none on the usage dump: reasoning_tokens was not reported here
        # and must not be stored as a fake zero.
        assert meta["usage"] == {"prompt_tokens": 3989, "completion_tokens": 708,
                                 "total_tokens": 4697}

    def test_contract_attached_flag_is_recorded_both_ways(self):
        assert _response_meta(_FakeMetaResponse(), True)["contract_attached"] is True
        assert _response_meta(_FakeMetaResponse(), False)["contract_attached"] is False

    def test_a_refusal_is_captured_as_text(self):
        """A strict-mode refusal is a contract outcome, not a crash, and it has
        to be counted separately from a schema violation."""
        meta = _response_meta(
            _FakeMetaResponse(finish_reason="stop", refusal="δεν μπορώ"), True)

        assert meta["refusal"] == "δεν μπορώ"

    def test_non_scalar_metadata_is_stringified(self):
        """`provider_specific_fields` is provider-shaped and can hold anything.
        It is coerced to a string so provenance.json always serialises."""
        meta = _response_meta(
            _FakeMetaResponse(provider_specific_fields={"grounding": True}), True)

        assert isinstance(meta["provider_specific_fields"], str)
        assert "grounding" in meta["provider_specific_fields"]

    def test_response_missing_every_optional_attribute_does_not_raise(self):
        """Metadata is best effort: a provider that returns a bare object must
        cost the analyst some columns, never the conversation."""
        meta = _response_meta(object(), False)

        assert meta == {"contract_attached": False}

    def test_usage_without_model_dump_is_still_captured(self):
        """Other providers hand back a plain mapping instead of a pydantic model;
        `dict(usage)` is the fallback path."""
        resp = _FakeMetaResponse()
        resp.usage = {"prompt_tokens": 10, "completion_tokens": 2}

        meta = _response_meta(resp, False)

        assert meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 2}


class TestLastTurnMeta:
    def test_one_entry_per_completion_call_of_the_turn(
            self, make_agent, monkeypatch, stub_run_tool):
        _install_recording_litellm(monkeypatch, [
            _tool_call_response(), make_response(content='{"status": "ok"}')])
        wfeds_agent = make_agent("structured")

        wfeds_agent.chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])

        assert len(wfeds_agent.last_turn_meta) == 2
        # The flag is per CALL, which is what lets the audit trail show that
        # round 1 went out uncontracted even in the treatment arm.
        assert wfeds_agent.last_turn_meta[0]["contract_attached"] is False
        assert wfeds_agent.last_turn_meta[1]["contract_attached"] is True

    def test_reset_at_the_start_of_each_turn(self, make_agent, monkeypatch,
                                             stub_run_tool):
        """Per TURN, not per conversation: provenance.json is written once per
        turn, so metadata from an earlier turn must not be attributed to this
        run's calls."""
        _install_recording_litellm(monkeypatch, [
            _tool_call_response(), make_response(content='{"status": "ok"}'),
            make_response(content="μια χαρά")])
        wfeds_agent = make_agent("structured")

        wfeds_agent.chat("τρέξε 6 ώρες", pins=[(38.9, 23.1)])
        assert len(wfeds_agent.last_turn_meta) == 2

        wfeds_agent.chat("ευχαριστώ")

        assert len(wfeds_agent.last_turn_meta) == 1
        assert wfeds_agent.last_turn_meta[0]["contract_attached"] is False
