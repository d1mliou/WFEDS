"""Tests for scripts/agent/telegram_bot.py.

Covers `_km_between` and `_n_fires` (the pure geometry helpers), plus
`on_location`'s geometry note and stale-pin distance warning -- driven with
`unittest.mock.MagicMock`/`AsyncMock` test doubles for the `Update`/context
objects (no `pytest-asyncio` dependency; the async handler is driven
synchronously via `asyncio.run(...)` inside plain `def test_...()` functions).
`telegram_bot.pins`/`.agents` are reset before/after every test in this
directory by the autouse fixture in tests/agent/conftest.py.
"""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

import telegram_bot
from build_cell2fire_instance import VIIRS_PIXEL_M, WINDOW_KM_MAX
from telegram_bot import _km_between, _n_fires


class TestKmBetween:
    def test_same_point_is_zero(self):
        p = (38.9, 23.1)
        assert _km_between(p, p) == pytest.approx(0.0)

    def test_one_degree_latitude_is_111_32_km(self):
        """Verified directly against the `111_320.0` m/deg-latitude constant the
        function itself uses (111_320.0 / 1000 = 111.32 km); at the equator the
        longitude component is zero, so this isolates the latitude term."""
        result = _km_between((0.0, 0.0), (1.0, 0.0))
        assert result == pytest.approx(111.32, abs=1e-6)

    def test_longitude_distance_scales_by_cos_latitude(self):
        """One degree of longitude is ~111.32 km at the equator but shrinks by
        cos(lat) at higher latitudes (meridians converge toward the poles)."""
        at_equator = _km_between((0.0, 0.0), (0.0, 1.0))
        at_60n = _km_between((60.0, 0.0), (60.0, 1.0))

        assert at_equator == pytest.approx(111.32, abs=1e-6)
        assert at_60n == pytest.approx(at_equator * math.cos(math.radians(60.0)),
                                       rel=1e-6)
        assert at_60n < at_equator

    def test_symmetric_in_its_two_arguments(self):
        p1, p2 = (38.90, 23.10), (38.95, 23.20)
        assert _km_between(p1, p2) == pytest.approx(_km_between(p2, p1))

    def test_combined_lat_lon_offset_uses_pythagorean_distance(self):
        """A pure-latitude offset and a pure-longitude offset of equal individual
        distance, combined, should give ~sqrt(2) times either alone (flat-Earth
        Pythagorean approximation, as the docstring says). Not EXACTLY sqrt(2):
        the longitude term is scaled by cos() of the MIDPOINT latitude, which for
        the combined (0,0)->(1,1) call is 0.5 deg (vs. 0 deg for the isolated
        lon_only call), a tiny real difference -- hence the looser tolerance."""
        lat_only = _km_between((0.0, 0.0), (1.0, 0.0))
        lon_only = _km_between((0.0, 0.0), (0.0, 1.0))
        combined = _km_between((0.0, 0.0), (1.0, 1.0))

        assert lat_only == pytest.approx(lon_only, abs=1e-6)   # both ~111.32 km at equator
        assert combined == pytest.approx(lat_only * math.sqrt(2), rel=1e-4)


class TestNFires:
    """`_n_fires` must mirror the engine's join rule: pin centres <= 375 m
    (VIIRS_PIXEL_M) merge into one front, transitively."""

    def test_single_pin_is_one_fire(self):
        assert _n_fires([(38.90, 23.10)], VIIRS_PIXEL_M) == 1

    def test_two_close_pins_merge_into_one_front(self):
        a, b = (38.90000, 23.10000), (38.90200, 23.10000)   # ~222 m apart
        assert _km_between(a, b) * 1000 <= VIIRS_PIXEL_M    # sanity-check the pair
        assert _n_fires([a, b], VIIRS_PIXEL_M) == 1

    def test_two_far_pins_stay_separate_fires(self):
        a, b = (38.90, 23.10), (38.91, 23.10)               # ~1.1 km apart
        assert _km_between(a, b) * 1000 > VIIRS_PIXEL_M
        assert _n_fires([a, b], VIIRS_PIXEL_M) == 2

    def test_chain_of_close_pins_merges_transitively(self):
        """a-b and b-c are each ~300 m (joinable) but a-c is ~600 m (not):
        union-find must still give ONE fire via the chain, like the engine's
        geometric union does."""
        a = (38.90000, 23.10000)
        b = (38.90270, 23.10000)
        c = (38.90540, 23.10000)
        assert _km_between(a, b) * 1000 <= VIIRS_PIXEL_M
        assert _km_between(a, c) * 1000 > VIIRS_PIXEL_M
        assert _n_fires([a, b, c], VIIRS_PIXEL_M) == 1


def _fake_update(lat, lon, chat_id=12345):
    """A MagicMock Update/message with the attribute chain on_location reads,
    plus an AsyncMock reply_text (a plain MagicMock would make `await
    fake.message.reply_text(...)` raise TypeError -- it can't be awaited)."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.location.latitude = lat
    update.message.location.longitude = lon
    update.message.reply_text = AsyncMock()
    return update


class TestOnLocationStalePinWarning:
    def test_first_pin_ever_sends_only_the_confirmation(self):
        update = _fake_update(38.90, 23.10)

        asyncio.run(telegram_bot.on_location(update, MagicMock()))

        assert update.message.reply_text.call_count == 1

    def test_second_pin_far_away_sends_a_second_separate_warning(self):
        p1 = (38.90, 23.10)
        p2 = (38.90 + 0.5, 23.10)   # ~0.5 deg latitude ~= 55.7 km
        assert _km_between(p1, p2) > WINDOW_KM_MAX   # sanity-check the fixture pair
        chat_id = 999
        telegram_bot.pins[chat_id] = [p1]
        update = _fake_update(*p2, chat_id=chat_id)

        asyncio.run(telegram_bot.on_location(update, MagicMock()))

        assert update.message.reply_text.call_count == 2
        warning_text = update.message.reply_text.call_args_list[1].args[0]
        assert "km" in warning_text
        assert f"{WINDOW_KM_MAX:.0f}" in warning_text

    def test_second_pin_close_sends_no_warning(self):
        p1 = (38.90, 23.10)
        p2 = (38.90 + 0.05, 23.10)   # ~5.6 km
        assert _km_between(p1, p2) < WINDOW_KM_MAX   # sanity-check the fixture pair
        chat_id = 888
        telegram_bot.pins[chat_id] = [p1]
        update = _fake_update(*p2, chat_id=chat_id)

        asyncio.run(telegram_bot.on_location(update, MagicMock()))

        assert update.message.reply_text.call_count == 1


class TestOnLocationGeometryNote:
    """The confirmation message must state how the ENGINE will read the pins
    (one merged front vs. N separate fires) at pin time - in the field test the
    user was told '4 pins = μέτωπο' and the run produced 4 independent fires."""

    def test_close_second_pin_reports_one_merged_front(self):
        chat_id = 777
        telegram_bot.pins[chat_id] = [(38.90000, 23.10000)]
        update = _fake_update(38.90200, 23.10000, chat_id=chat_id)   # ~222 m

        asyncio.run(telegram_bot.on_location(update, MagicMock()))

        text = update.message.reply_text.call_args_list[0].args[0]
        assert "ένα ενιαίο μέτωπο" in text

    def test_far_second_pin_reports_two_separate_fires(self):
        chat_id = 666
        telegram_bot.pins[chat_id] = [(38.90, 23.10)]
        update = _fake_update(38.91, 23.10, chat_id=chat_id)         # ~1.1 km

        asyncio.run(telegram_bot.on_location(update, MagicMock()))

        assert update.message.reply_text.call_count == 1   # within the window: no stale-pin warning
        text = update.message.reply_text.call_args_list[0].args[0]
        assert "2 ξεχωριστές φωτιές" in text
        assert "ενδιάμεσα pins" in text
