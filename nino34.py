import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

JSON_URL = (
    "https://climatereanalyzer.org/clim/sst_daily/json_2clim/"
    "oisst2.1_nino3.4_sst_day.json"
)
JSON_FILE = Path("oisst2.1_nino3.4_sst_day.json")
OUTPUT = Path("nino34_daily_sst_anomaly_1991_2020.png")
BASELINE_NAME = "1991-2020"

def download_json(url: str, destination: Path) -> None:
    """Download the source JSON using browser-like headers required by the site."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/150.0.0.0 Safari/537.36"
        ),
        "Referer": "https://climatereanalyzer.org/clim/sst_daily/",
        "Accept": "application/json,text/plain,*/*",
    }
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


download_json(JSON_URL, JSON_FILE)

with JSON_FILE.open(encoding="utf-8") as f:
    records = json.load(f)

series = {
    item["name"]: np.array(
        [np.nan if value is None else float(value) for value in item["data"]],
        dtype=float,
    )
    for item in records
}

if BASELINE_NAME not in series:
    raise KeyError(f"Baseline {BASELINE_NAME!r} is not present in the JSON")

baseline = series[BASELINE_NAME]
anomalies = {
    name: values - baseline
    for name, values in series.items()
    if name not in ("1982-2010", BASELINE_NAME, "Preliminary")
}

# Preliminary values continue the finalized 2026 series in subsequent day slots.
anomaly_2026_final = series["2026"] - baseline
preliminary = series["Preliminary"]
anomaly_preliminary = preliminary - baseline

# Include the final finalized point at the start of the orange segment so the
# finalized and preliminary lines join without a visual gap.
preliminary_plot = anomaly_preliminary.copy()
first_preliminary_idx = np.flatnonzero(~np.isnan(anomaly_preliminary))[0]
preliminary_plot[first_preliminary_idx - 1] = anomaly_2026_final[first_preliminary_idx - 1]

# A non-leap reference converts the supplied zero-based day indices to month labels.
dates = np.array([datetime(2025, 1, 1) + timedelta(days=i) for i in range(366)])

fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)

years = sorted(int(name) for name in anomalies if name.isdigit())
highlight_years = {
    1982: "#6a4c93",
    1997: "#2a9d8f",
    2015: "#2374ab",
}
for year in years:
    if year in (*highlight_years, 2026):
        continue
    ax.plot(
        dates,
        anomalies[str(year)],
        color="#7d8790",
        lw=0.7,
        alpha=0.38,
        zorder=1,
    )

ax.axhline(0, color="#222222", lw=1.8, linestyle="--", zorder=3)

for year, color in highlight_years.items():
    ax.plot(
        dates,
        anomalies[str(year)],
        color=color,
        lw=2.0,
        label=str(year),
        zorder=4,
    )

ax.plot(
    dates,
    anomaly_2026_final,
    color="#d1495b",
    lw=2.8,
    label="2026",
    zorder=5,
)
ax.plot(
    dates,
    preliminary_plot,
    color="#f28e2b",
    lw=2.8,
    label="Preliminary 2026",
    zorder=6,
)

last_idx = np.flatnonzero(~np.isnan(anomaly_preliminary))[-1]
last_date = dates[last_idx]
last_value = anomaly_preliminary[last_idx]

# Find the highest anomaly measured in another year on the same day.
other_year_values = {
    year: anomalies[str(year)][last_idx]
    for year in years
    if year != 2026 and not np.isnan(anomalies[str(year)][last_idx])
}

highest_other_year = max(
    other_year_values,
    key=other_year_values.get,
)
highest_other_value = other_year_values[highest_other_year]
difference_from_highest = last_value - highest_other_value

ax.scatter(
    last_date,
    last_value,
    s=45,
    color="#f28e2b",
    edgecolor="white",
    linewidth=0.8,
    zorder=7,
)

ax.annotate(
    (
        f"{last_date.day} {last_date:%b}\n"
        f"2026: {last_value:+.2f} \u00b0C\n"
        f"Previous maximum: {highest_other_value:+.2f} \u00b0C "
        f"({highest_other_year})\n"
        f"Difference: {difference_from_highest:+.2f} \u00b0C"
    ),
    xy=(last_date, last_value),
    xycoords="data",
    xytext=(0.99, 1.05),
    textcoords="axes fraction",
    fontsize=9,
    ha="right",
    va="center",
    bbox={
        "boxstyle": "round,pad=0.4",
        "facecolor": "white",
        "edgecolor": "#f28e2b",
        "alpha": 0.95,
    },
    arrowprops={
        "arrowstyle": "-",
        "color": "#f28e2b",
        "lw": 1,
    },
    annotation_clip=False,
    clip_on=False,
    zorder=8,
)

ax.set_title(
    "Daily Niño 3.4 sea-surface temperature anomaly",
    fontsize=18,
    weight="bold",
    loc="left",
    pad=32,
)
ax.text(
    0,
    1.01,
    "Relative to the 1991–2020 daily climatology · Other historical years in grey (1981–2026 available)",
    transform=ax.transAxes,
    fontsize=10.5,
    color="#555555",
    va="bottom",
)
ax.set_ylabel("Temperature anomaly (°C)")
ax.set_xlabel("Day of year")
ax.set_xlim(dates[0], dates[364])
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
ax.grid(axis="x", visible=False)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.15))
ax.text(
    1,
    -0.12,
    "Data source: Climate Reanalyzer / NOAA OISST 2.1",
    transform=ax.transAxes,
    fontsize=8.5,
    color="#666666",
    ha="right",
)

fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Latest 2026 anomaly: {last_value:+.3f} °C on 2026-{last_date:%m-%d}")
print(OUTPUT.resolve())

# ---------------------------------------------------------------------------
# Second graph: difference between 2026 and the maximum of all other years
# ---------------------------------------------------------------------------

# Stack all annual anomaly series except 2026.
other_year_anomalies = np.vstack(
    [
        anomalies[str(year)]
        for year in years
        if year != 2026
    ]
)

# Highest anomaly recorded on each calendar day in all other years.
maximum_other_years = np.nanmax(other_year_anomalies, axis=0)

# Calculate the difference separately for finalized and preliminary 2026 data.
difference_2026_final = anomaly_2026_final - maximum_other_years
difference_2026_preliminary = anomaly_preliminary - maximum_other_years

# Include the last finalized point at the start of the preliminary segment.
difference_preliminary_plot = difference_2026_preliminary.copy()
difference_preliminary_plot[first_preliminary_idx - 1] = (
    difference_2026_final[first_preliminary_idx - 1]
)

fig_difference, ax_difference = plt.subplots(
    figsize=(14, 7),
    constrained_layout=True,
)

ax_difference.axhline(
    0,
    color="#222222",
    linewidth=1.8,
    linestyle="--",
    zorder=2,
)

ax_difference.plot(
    dates,
    difference_2026_final,
    color="#d1495b",
    linewidth=2.8,
    label="2026",
    zorder=3,
)

ax_difference.plot(
    dates,
    difference_preliminary_plot,
    color="#f28e2b",
    linewidth=2.8,
    label="Preliminary 2026",
    zorder=4,
)

# Shade days on which 2026 exceeds all other years.
combined_difference = difference_2026_final.copy()
preliminary_mask = ~np.isnan(difference_2026_preliminary)
combined_difference[preliminary_mask] = difference_2026_preliminary[
    preliminary_mask
]

ax_difference.fill_between(
    dates,
    0,
    combined_difference,
    where=combined_difference > 0,
    color="#f28e2b",
    alpha=0.15,
    interpolate=True,
    label="2026 above previous maximum",
    zorder=1,
)

# Mark and annotate the most recent preliminary value.
latest_difference = difference_2026_preliminary[last_idx]

ax_difference.scatter(
    last_date,
    latest_difference,
    s=45,
    color="#f28e2b",
    edgecolor="white",
    linewidth=0.8,
    zorder=5,
)

ax_difference.annotate(
    (
        f"{last_date.day} {last_date:%b}\n"
        f"Difference: {latest_difference:+.2f} \u00b0C\n"
        f"({highest_other_year})"
    ),
    xy=(last_date, latest_difference),
    xycoords="data",
    xytext=(0.99, 1.14),
    textcoords="axes fraction",
    fontsize=9,
    ha="right",
    va="center",
    bbox={
        "boxstyle": "round,pad=0.4",
        "facecolor": "white",
        "edgecolor": "#f28e2b",
        "alpha": 0.95,
    },
    arrowprops={
        "arrowstyle": "-",
        "color": "#f28e2b",
        "linewidth": 1,
    },
    annotation_clip=False,
    clip_on=False,
    zorder=6,
)

ax_difference.set_title(
    "Difference between 2026 Niño 3.4 SST and previous daily maximum",
    fontsize=18,
    weight="bold",
    loc="left",
    pad=32,
)

ax_difference.text(
    0,
    1.01,
    (
        "2026 minus the highest value measured on the same calendar day "
        "in any other available year"
    ),
    transform=ax_difference.transAxes,
    fontsize=10.5,
    color="#555555",
    va="bottom",
)

ax_difference.set_ylabel("Difference from previous maximum (\u00b0C)")
ax_difference.set_xlabel("Day of year")
ax_difference.set_xlim(dates[0], dates[364])

ax_difference.xaxis.set_major_locator(mdates.MonthLocator())
ax_difference.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

ax_difference.grid(
    axis="y",
    color="#d9d9d9",
    linewidth=0.8,
)
ax_difference.grid(axis="x", visible=False)

ax_difference.spines[["top", "right"]].set_visible(False)

ax_difference.legend(
    frameon=False,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.17),
)

ax_difference.text(
    1,
    -0.14,
    "Data source: Climate Reanalyzer / NOAA OISST 2.1",
    transform=ax_difference.transAxes,
    fontsize=8.5,
    color="#666666",
    ha="right",
)

DIFFERENCE_OUTPUT = Path(
    "nino34_2026_difference_from_previous_daily_maximum.png"
)

fig_difference.savefig(
    DIFFERENCE_OUTPUT,
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)

plt.close()

print(DIFFERENCE_OUTPUT.resolve())