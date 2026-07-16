"""Canadian Fire Weather Index (FWI) System codes from REAL weather.

Replaces the frozen PLACEHOLDER codes in `Weather.csv` (FFMC 90.55 / ISI 13.35 etc.,
copied from a Cell2Fire example) that made the fuel permanently bone-dry and the fire
insensitive to temperature/humidity - the root cause of the unrealistic spread rate
(2026-07-02, see [[Decision log]]).

Standard daily equations (Van Wagner 1987; Van Wagner & Pickett 1985):
  * FFMC / DMC / DC update once per DAY from noon weather + 24 h rain, carrying
    yesterday's codes -> a SPIN-UP over the preceding weeks initialises them
    (defaults FFMC 85 / DMC 6 / DC 15; DC has ~50-day memory, so start >= 1 month
    before the event - we use June 1).
  * "Noon" = 12:00 local standard time; for Evia (23.3E, UTC+2 LST) we take the
    10:00 UTC hourly values; rain = the 24 h sum ending that hour.
  * ISI is recomputed PER HOUR from the day's FFMC + that hour's wind speed (the
    standard way to drive hourly fire behaviour), BUI from DMC/DC, FWI from ISI+BUI.

Pure module - no I/O; `build_cell2fire_instance.py` feeds it Open-Meteo rows.
"""

from __future__ import annotations

import math

# Day-length adjustment tables, by calendar month (Jan..Dec) - the Canadian system's
# way of accounting for how much daylight/evaporation a given month gets, so the
# SAME weather dries fuel faster in June than in December.
DMC_LE = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]   # Duff Moisture Code
DC_LF = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]    # Drought Code

FFMC_INIT, DMC_INIT, DC_INIT = 85.0, 6.0, 15.0        # spin-up starting codes (fresh-season defaults)


def daily_ffmc(f0, temp, rh, wind, rain):
    """Fine Fuel Moisture Code (yesterday f0 -> today), Van Wagner 1987 eq. 1-10."""
    mo = 147.2 * (101.0 - f0) / (59.5 + f0)          # yesterday's code -> moisture content (%)
    if rain > 0.5:                                    # rain event -> fuel WETS (absorbs moisture)
        rf = rain - 0.5                               # canopy/interception loss below 0.5 mm
        mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1 - math.exp(-6.93 / rf))
        if mo > 150.0:                                # already-saturated fuel absorbs even more
            mr += 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
        mo = min(mr, 250.0)                           # physical ceiling on fuel moisture
    ed = (0.942 * rh ** 0.679 + 11.0 * math.exp((rh - 100.0) / 10.0)         # drying equilibrium moisture
          + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh)))
    if mo > ed:                                       # fuel wetter than equilibrium -> DRYING branch
        ko = (0.424 * (1.0 - (rh / 100.0) ** 1.7)
              + 0.0694 * math.sqrt(wind) * (1.0 - (rh / 100.0) ** 8))        # log drying rate (RH + wind driven)
        kd = ko * 0.581 * math.exp(0.0365 * temp)                           # temperature-adjusted drying rate
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = (0.618 * rh ** 0.753 + 10.0 * math.exp((rh - 100.0) / 10.0)    # wetting equilibrium moisture
              + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh)))
        if mo < ew:                                   # fuel drier than equilibrium -> WETTING branch
            kl = (0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7)
                  + 0.0694 * math.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0) ** 8))   # log wetting rate
            kw = kl * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo                                    # already at equilibrium -> no change
    return 59.5 * (250.0 - m) / (147.2 + m)           # moisture (%) back to the FFMC code scale


def daily_dmc(p0, temp, rh, rain, month):
    """Duff Moisture Code, Van Wagner 1987 eq. 11-17."""
    if rain > 1.5:                                    # enough rain to wet the duff layer (deeper than FFMC's)
        re = 0.92 * rain - 1.27                       # effective rain reaching the duff (canopy loss)
        mo = 20.0 + math.exp(5.6348 - p0 / 43.43)     # yesterday's code -> duff moisture content
        if p0 <= 33.0:                                # slope of the moisture/rain response, by dryness class
            b = 100.0 / (0.5 + 0.3 * p0)
        elif p0 <= 65.0:
            b = 14.0 - 1.3 * math.log(p0)
        else:
            b = 6.2 * math.log(p0) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)      # moisture after absorbing today's rain
        p0 = max(0.0, 43.43 * (5.6348 - math.log(mr - 20.0)))   # back to the DMC scale (this is the rain-adjusted start)
    k = (1.894 * (temp + 1.1) * (100.0 - rh) * DMC_LE[month - 1] * 1e-6     # today's drying, day-length weighted
         if temp > -1.1 else 0.0)                     # too cold -> no drying at all
    return p0 + 100.0 * k


def daily_dc(d0, temp, rain, month):
    """Drought Code, Van Wagner 1987 eq. 18-22."""
    if rain > 2.8:                                    # enough rain to recharge the deep/soil moisture layer
        rd = 0.83 * rain - 1.27                       # effective rain reaching that depth
        qo = 800.0 * math.exp(-d0 / 400.0)            # yesterday's code -> moisture equivalent
        qr = qo + 3.937 * rd                          # after today's recharge
        d0 = max(0.0, 400.0 * math.log(800.0 / qr))   # back to the DC scale
    v = max(0.0, 0.36 * (temp + 2.8) + DC_LF[month - 1]) if temp > -2.8 else \
        max(0.0, DC_LF[month - 1])                    # today's drying potential (day-length weighted, floored at 0)
    return d0 + 0.5 * v


def isi(ffmc, wind):
    """Initial Spread Index from FFMC + wind (km/h) - recompute PER HOUR."""
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)        # FFMC -> fine fuel moisture content
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)   # fine-fuel ignition/spread function
    return 0.208 * math.exp(0.05039 * wind) * ff      # wind multiplier (exponential) applied on top


def bui(dmc, dc):
    """Buildup Index from DMC + DC."""
    if dmc <= 0.4 * dc:                               # duff moisture dominates -> simple blend
        return 0.8 * dmc * dc / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0.0
    return dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)   # DC-dominated correction


def fwi(isi_, bui_):
    """Fire Weather Index from ISI + BUI."""
    fd = (0.626 * bui_ ** 0.809 + 2.0 if bui_ <= 80.0             # fuel-availability function of BUI
          else 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui_)))   # saturates at very high BUI (extreme drought)
    b = 0.1 * isi_ * fd                               # intermediate combination of spread + fuel availability
    return math.exp(2.72 * (0.434 * math.log(b)) ** 0.647) if b > 1.0 else b   # log/exp scaling only kicks in above 1


def spinup_daily_codes(hourly_rows, noon_hour_utc=10):
    """Hourly Open-Meteo rows (weeks!) -> {date: (ffmc, dmc, dc)} via the daily system.

    Noon inputs = the `noon_hour_utc` row of each date; rain = 24 h precipitation sum
    ending at that hour. Days before the first full day are skipped."""
    by_time = {r["time"]: r for r in hourly_rows}     # index the hourly series by ISO timestamp
    times = sorted(by_time)
    dates = sorted({t[:10] for t in times})           # the distinct calendar dates covered
    f, p, d = FFMC_INIT, DMC_INIT, DC_INIT            # carried day-to-day, starting from the spin-up defaults
    out = {}
    for date in dates:
        key = f"{date}T{noon_hour_utc:02d}:00"        # this date's "noon" row (10:00 UTC = local noon for Evia)
        if key not in by_time:
            continue                                  # partial first/last day -> skip, no noon reading
        noon = by_time[key]
        idx = times.index(key)
        rain = sum((by_time[t]["precip_mm"] or 0.0) for t in times[max(0, idx - 23):idx + 1])   # 24h rain ending at noon
        month = int(date[5:7])                        # for the day-length tables above
        temp, rh, wind = noon["temp_c"], noon["rh_pct"], noon["ws_kmh"]
        if temp is None or rh is None or wind is None:
            continue                                  # missing noon reading -> can't update this day, carry forward
        f = daily_ffmc(f, temp, rh, wind, rain)        # each code updates from YESTERDAY's own value (the memory)
        p = daily_dmc(p, temp, rh, rain, month)
        d = daily_dc(d, temp, rain, month)
        out[date] = (round(f, 2), round(p, 2), round(d, 2))
    return out


if __name__ == "__main__":
    # self-check: hot, dry, windy summer day on top of dry codes -> extreme indices
    f, p, d = 92.0, 120.0, 500.0
    i = isi(f, 30.0)
    b = bui(p, d)
    print(f"FFMC {f} + 30 km/h -> ISI {i:.1f} | BUI {b:.1f} | FWI {fwi(i, b):.1f}")
    f2 = daily_ffmc(85.0, 32.0, 20.0, 15.0, 0.0)
    print(f"one hot dry day from FFMC 85 -> {f2:.1f} (should rise toward ~90)")
