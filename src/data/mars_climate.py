from __future__ import annotations

from dataclasses import dataclass
import csv
import importlib
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from scipy.io import netcdf_file


MARS_YEAR_SOLS = 669.6
MARS_SOLAR_CONSTANT_W_M2 = 586.2

#: Eccentricita' dell'orbita di Marte e longitudine areocentrica del perielio.
#: Sono costanti astronomiche pubblicate, non parametri di taratura.
MARS_ECCENTRICITY = 0.0934
MARS_LS_PERIHELION_DEG = 250.99

#: Longitudine areocentrica attorno a cui culmina la stagione delle tempeste di
#: polvere marziane.
#:
#: **Perche' esiste questa costante (2026-08-30).** Il ciclo stagionale della
#: polvere era `(0,5 + 0,5 sin Ls)^4`, cioe' con il **massimo a Ls 90** e lo
#: **zero a Ls 251**. E' l'antifase esatta della climatologia marziana: Ls 71-90
#: e' la stagione limpida attorno all'afelio, mentre le tempeste -- comprese
#: tutte le tempeste globali storiche -- cadono nella primavera-estate australe,
#: fra Ls 180 e Ls 330, attorno al perielio. Il modello metteva cielo terso nel
#: pieno della stagione polverosa e viceversa.
#:
#: Il valore non e' tarato: e' il centro della stagione osservata, e coincide
#: con il perielio entro un grado.
MARS_LS_DUST_PEAK_DEG = 250.0


def fattore_insolazione_orbitale(ls_deg: float) -> float:
    """Quante volte l'insolazione media vale a quella longitudine areocentrica.

    **Perche' serve (2026-08-30).** Il flusso solare pubblicato valeva
    `586,2 x (1 - 0,45 tau)`, cioe' una funzione della **sola polvere**:
    verificato invertendolo su dieci campioni, il tau ricavato coincideva
    esattamente con l'opacita' climatica. Non conteneva alcuna informazione
    solare, e siccome `planetary_coupling` lo usava **accanto** alla polvere per
    scaldare e irradiare le celle, la polvere entrava due volte — il 41,8%
    dell'effetto termico arrivava da un canale chiamato «solare».

    Con l'eccentricita' reale il flusso varia del 31% lungo l'anno, e la
    formula riproduce l'intervallo pubblicato per Marte: **490,3 - 713,2 W/m2**
    contro i 493 - 717 misurati. Nessun numero e' stato scelto: sono
    l'eccentricita' e il perielio di Marte, e la legge di Keplero.
    """
    import math

    anomalia = math.radians(float(ls_deg) - MARS_LS_PERIHELION_DEG)
    distanza_relativa = (1.0 - MARS_ECCENTRICITY ** 2) / (
        1.0 + MARS_ECCENTRICITY * math.cos(anomalia)
    )
    return float(1.0 / (distanza_relativa ** 2))
MCD_SCENARIO_ARCHIVES = {
    "climatology": ("MCD_6.1.tar.gz", "MCD6.1.tar.gz", "MCD_pr 6.1.tar.gz"),
    "clim_minEUV": ("clim_minEUV.tar.gz",),
    "clim_maxEUV": ("clim_maxEUV.tar.gz",),
    "cold": ("cold.tar.gz",),
    "warm": ("warm.tar.gz",),
    "strm": ("strm.tar.gz",),
    "dust_storm": ("strm.tar.gz",),
}


@dataclass(frozen=True)
class MarsClimateSample:
    pressure_pa: float
    mean_temperature_c: float
    dust_opacity: float
    solar_flux_w_m2: float
    relative_humidity: float
    wind_speed_m_s: float
    source: str
    scenario: str
    data_path: str
    data_available: bool
    sol: float
    ls_deg: float

    @property
    def solar_flux_toa_w_m2(self) -> float:
        """Il flusso al top dell'atmosfera: SOLO orbita, nessuna polvere.

        `solar_flux_w_m2` resta il flusso attenuato dalla polvere ed e' quello
        giusto per chi vuole l'energia che arriva a terra. Questo e' quello
        giusto per chi vuole l'insolazione come **driver indipendente** dalla
        polvere, e sono due grandezze diverse che non vanno confuse.
        """
        return MARS_SOLAR_CONSTANT_W_M2 * fattore_insolazione_orbitale(self.ls_deg)

    def to_metrics(self) -> dict[str, float | str | bool]:
        return {
            "climate_pressure_pa": self.pressure_pa,
            "climate_mean_temperature_c": self.mean_temperature_c,
            "climate_dust_opacity": self.dust_opacity,
            "climate_solar_flux_w_m2": self.solar_flux_w_m2,
            "climate_solar_flux_toa_w_m2": self.solar_flux_toa_w_m2,
            "climate_relative_humidity": self.relative_humidity,
            "climate_wind_speed_m_s": self.wind_speed_m_s,
            "climate_source": self.source,
            "climate_scenario": self.scenario,
            "climate_data_path": self.data_path,
            "climate_data_available": self.data_available,
            "climate_sol": self.sol,
            "climate_ls_deg": self.ls_deg,
        }


class MarsClimateProvider:
    """
    Small, dependency-light climate adapter.

    It can read a local CSV sample table now, detects a downloaded MCD archive for audit,
    and otherwise falls back to a documented MCD-style climatology proxy based on seasonal
    pressure, dust and temperature cycles.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("climate", {}) if isinstance((config or {}).get("climate", {}), dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.source = str(cfg.get("source", "auto"))
        self.provider = str(cfg.get("provider", "auto")).lower()
        self.scenario = str(cfg.get("scenario", "climatology"))
        self.landing_site = str(cfg.get("landing_site", "mid_latitude"))
        self.latitude_deg = float(cfg.get("latitude_deg", -4.6))
        self.longitude_deg = float(cfg.get("longitude_deg", 137.4))
        self.local_hour = float(cfg.get("local_hour", 12.0))
        self.local_path = _resolve_path(cfg.get("mcd_path") or cfg.get("data_path") or os.environ.get("MCD_PATH"), self.scenario)
        self.available_archives = _detect_available_archives()
        self._csv_rows = self._load_csv_rows(self.local_path)
        self._netcdf_root = _find_netcdf_root(self.local_path, self.scenario)
        self._archive_available = self._archive_has_climate_files(self.local_path)
        self._fmcd = _load_fmcd_module(self.local_path)
        self._mcd_dataset_path = _resolve_mcd_dataset_path(self.local_path, self.scenario)

    def sample(
        self,
        sol: float,
        latitude_deg: float = -4.6,
        elevation_m: float = -2500.0,
        longitude_deg: float | None = None,
    ) -> MarsClimateSample:
        sample_longitude = self.longitude_deg if longitude_deg is None else float(longitude_deg)
        if not self.enabled:
            return _analytic_sample(sol, latitude_deg, elevation_m, "disabled", self.scenario, "", False)
        if self._csv_rows:
            return self._sample_csv(sol)
        if self.provider in {"call_mcd", "fortran", "mcd_fortran"} and self._fmcd and self._mcd_dataset_path:
            sample = _sample_call_mcd(
                self._fmcd,
                self._mcd_dataset_path,
                self.scenario,
                sol,
                self.latitude_deg if latitude_deg == -4.6 else latitude_deg,
                sample_longitude,
                self.local_hour,
                elevation_m,
            )
            if sample:
                return sample
        if self._netcdf_root:
            sample = _sample_netcdf(
                self._netcdf_root,
                self.scenario,
                sol,
                self.latitude_deg if latitude_deg == -4.6 else latitude_deg,
                sample_longitude,
                self.local_hour,
            )
            if sample:
                return sample
        if self.provider == "call_mcd":
            return _analytic_sample(sol, latitude_deg, elevation_m, "mcd_call_unavailable_proxy", self.scenario, str(self._mcd_dataset_path or self.local_path or ""), False)
        if self.local_path and self._archive_available:
            sample = _analytic_sample(sol, latitude_deg, elevation_m, "mcd_archive_detected_proxy", self.scenario, str(self.local_path), True)
            return sample
        return _analytic_sample(sol, latitude_deg, elevation_m, "mcd_analytic_proxy", self.scenario, str(self.local_path or ""), False)

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "scenario": self.scenario,
            "local_path": str(self.local_path or ""),
            "available_archives": self.available_archives,
            "csv_rows": len(self._csv_rows),
            "netcdf_root": str(self._netcdf_root or ""),
            "archive_available": self._archive_available,
            "provider": self.provider,
            "call_mcd_available": bool(self._fmcd and self._mcd_dataset_path),
            "mcd_dataset_path": str(self._mcd_dataset_path or ""),
        }

    def _load_csv_rows(self, path: Path | None) -> list[dict[str, float]]:
        if not path:
            return []
        csv_path = path if path.is_file() and path.suffix.lower() == ".csv" else path / "mars_climate_samples.csv"
        if not csv_path.exists():
            return []
        rows: list[dict[str, float]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append(
                        {
                            "sol": float(row.get("sol", row.get("mars_sol", 0.0))),
                            "pressure_pa": float(row.get("pressure_pa", row.get("surface_pressure_pa", 700.0))),
                            "mean_temperature_c": float(row.get("mean_temperature_c", row.get("temperature_c", -63.0))),
                            "dust_opacity": float(row.get("dust_opacity", row.get("tau", 0.18))),
                            "relative_humidity": float(row.get("relative_humidity", 0.0)),
                            "wind_speed_m_s": float(row.get("wind_speed_m_s", 0.0)),
                        }
                    )
                except ValueError:
                    continue
        return sorted(rows, key=lambda item: item["sol"])

    def _sample_csv(self, sol: float) -> MarsClimateSample:
        rows = self._csv_rows
        wrapped_sol = sol % MARS_YEAR_SOLS
        before = max((row for row in rows if row["sol"] <= wrapped_sol), default=rows[-1])
        after = min((row for row in rows if row["sol"] >= wrapped_sol), default=rows[0])
        if before is after:
            blended = before
        else:
            span = max(1e-9, after["sol"] - before["sol"])
            t = (wrapped_sol - before["sol"]) / span
            blended = {key: before[key] + (after[key] - before[key]) * t for key in before}
        return MarsClimateSample(
            pressure_pa=blended["pressure_pa"],
            mean_temperature_c=blended["mean_temperature_c"],
            dust_opacity=max(0.0, min(5.0, blended["dust_opacity"])),
            solar_flux_w_m2=MARS_SOLAR_CONSTANT_W_M2 * (1.0 - 0.45 * max(0.0, min(1.0, blended["dust_opacity"]))),
            relative_humidity=max(0.0, min(1.5, blended["relative_humidity"])),
            wind_speed_m_s=max(0.0, blended["wind_speed_m_s"]),
            source="csv_climate_table",
            scenario=self.scenario,
            data_path=str(self.local_path or ""),
            data_available=True,
            sol=wrapped_sol,
            ls_deg=_ls_deg(wrapped_sol),
        )

    def _archive_has_climate_files(self, path: Path | None) -> bool:
        if not path or not path.exists():
            return False
        if path.is_dir():
            return any(item.suffix.lower() in {".nc", ".cdf", ".dat"} for item in path.rglob("*") if item.is_file())
        if path.suffix.lower() == ".gz" and path.name.lower().endswith(".tar.gz"):
            # MCD archives can be multiple GB: opening them just to inspect members
            # makes app startup unacceptably slow. Existence is enough here; actual
            # extraction/parsing is handled by CSV/extracted-folder paths.
            return path.stat().st_size > 0
        return False


def _analytic_sample(sol: float, latitude_deg: float, elevation_m: float, source: str, scenario: str, data_path: str, data_available: bool) -> MarsClimateSample:
    wrapped_sol = sol % MARS_YEAR_SOLS
    ls_rad = 2.0 * math.pi * wrapped_sol / MARS_YEAR_SOLS
    latitude_factor = abs(latitude_deg) / 90.0
    pressure_base = 700.0 * math.exp(-elevation_m / 10800.0)
    pressure = pressure_base * (1.0 + 0.13 * math.sin(ls_rad - 0.45))
    dust = _scenario_dust(scenario, ls_rad)
    seasonal_temp = -63.0 + 18.0 * math.sin(ls_rad - 0.25) - 24.0 * latitude_factor
    dust_warming = min(10.0, dust * 6.0)
    temp = seasonal_temp + dust_warming
    rh = max(0.02, min(1.0, 0.22 + 0.38 * latitude_factor + 0.12 * math.cos(ls_rad)))
    wind = max(0.0, 6.0 + 4.0 * dust + 2.0 * math.sin(ls_rad * 2.0))
    return MarsClimateSample(
        pressure_pa=pressure,
        mean_temperature_c=temp,
        dust_opacity=dust,
        solar_flux_w_m2=MARS_SOLAR_CONSTANT_W_M2 * (1.0 - 0.45 * min(1.0, dust)),
        relative_humidity=rh,
        wind_speed_m_s=wind,
        source=source,
        scenario=scenario,
        data_path=data_path,
        data_available=data_available,
        sol=wrapped_sol,
        ls_deg=_ls_deg(wrapped_sol),
    )


def _scenario_dust(scenario: str, ls_rad: float) -> float:
    # Il picco cade nella stagione polverosa australe, non nella limpida
    # settentrionale: vedi `MARS_LS_DUST_PEAK_DEG` per la misura che lo impone.
    # Lo sfasamento porta il massimo del seno su `MARS_LS_DUST_PEAK_DEG`.
    sfasamento = math.radians(MARS_LS_DUST_PEAK_DEG - 90.0)
    seasonal = (0.5 + 0.5 * math.sin(ls_rad - sfasamento)) ** 4
    scenario = scenario.lower()
    if scenario in {"dust_storm", "storm", "strm"}:
        return min(5.0, 1.4 + 3.6 * seasonal)
    if scenario in {"warm", "dusty", "clim_maxeuv"}:
        return min(1.2, 0.22 + 0.58 * seasonal)
    if scenario in {"cold", "clear", "clim_mineuv"}:
        return min(0.28, 0.035 + 0.10 * seasonal)
    return min(0.85, 0.06 + 0.18 * seasonal)


def _resolve_path(raw: Any, scenario: str = "climatology") -> Path | None:
    if not raw:
        candidates = [Path("data/mcd_runtime"), Path("data/mcd_windows"), Path("data/mcd")]
        names = MCD_SCENARIO_ARCHIVES.get(scenario, ()) + MCD_SCENARIO_ARCHIVES["climatology"]
        roots = [Path("data"), Path("."), Path.home() / "Downloads", Path.home() / "Desktop"]
        candidates.extend(root / name for root in roots for name in names)
        return next((candidate for candidate in candidates if candidate.exists()), None)
    return Path(str(raw)).expanduser()


def _ls_deg(sol: float) -> float:
    return (sol % MARS_YEAR_SOLS) / MARS_YEAR_SOLS * 360.0


def _detect_available_archives() -> list[str]:
    available: list[str] = []
    roots = [Path("data"), Path("."), Path.home() / "Downloads", Path.home() / "Desktop"]
    for scenario, names in MCD_SCENARIO_ARCHIVES.items():
        if scenario == "dust_storm":
            continue
        if any((root / name).exists() for root in roots for name in names):
            available.append(scenario)
    return sorted(set(available))


def _find_netcdf_root(path: Path | None, scenario: str) -> Path | None:
    if not path or not path.exists() or not path.is_dir():
        return None
    candidates = [path, path / "data", path / "MCD_6.1" / "data"]
    folder = _scenario_folder(scenario)
    for root in candidates:
        scenario_dir = root / folder
        if scenario_dir.exists() and any(scenario_dir.glob("*.nc")):
            return scenario_dir
        if any(root.glob("*.nc")):
            return root
    return None


def _scenario_folder(scenario: str) -> str:
    return {
        "climatology": "clim_aveEUV",
        "clim_minEUV": "clim_minEUV",
        "clim_maxEUV": "clim_maxEUV",
        "cold": "cold",
        "warm": "warm",
        "strm": "strm",
        "dust_storm": "strm",
    }.get(scenario, scenario)


def _scenario_file_prefix(scenario: str) -> str:
    if scenario in {"climatology", "clim_minEUV", "clim_maxEUV"}:
        return "clim"
    if scenario == "dust_storm":
        return "strm"
    return scenario


def _sample_netcdf(root: Path, scenario: str, sol: float, latitude_deg: float, longitude_deg: float, local_hour: float) -> MarsClimateSample | None:
    month = int((sol % MARS_YEAR_SOLS) / MARS_YEAR_SOLS * 12) + 1
    month = max(1, min(12, month))
    prefix = _scenario_file_prefix(scenario)
    nc_path = _resolve_monthly_netcdf(root, prefix, month)
    if not nc_path:
        return None
    try:
        with netcdf_file(nc_path, "r", mmap=False) as nc:
            lat = np.array(nc.variables["latitude"].data, dtype=float)
            lon = np.array(nc.variables["longitude"].data, dtype=float)
            time = np.array(nc.variables["Time"].data, dtype=float)
            lat_i = int(np.abs(lat - latitude_deg).argmin())
            lon_i = int(np.abs(((lon - (longitude_deg % 360.0) + 180.0) % 360.0) - 180.0).argmin())
            # The netcdf Time axis is universal time: convert each sample to the
            # local hour at the sampled longitude (15 deg east = +1 h) and match
            # on the 24-hour circle, so noon is noon at every landing site.
            lon_east = ((longitude_deg + 180.0) % 360.0) - 180.0
            hour_values = (time + lon_east / 15.0) % 24.0
            hour_distance = np.abs(((hour_values - (local_hour % 24.0)) + 12.0) % 24.0 - 12.0)
            time_i = int(hour_distance.argmin()) if time.size else 0
            ps = _read_var(nc, "ps", (time_i, lat_i, lon_i), 700.0)
            tsurf_k = _read_var(nc, "tsurf", (time_i, lat_i, lon_i), 210.0)
            tau = _read_var(nc, "tau_pref_gcm", (time_i, lat_i, lon_i), 0.18)
            # Solar flux is used for energy budgets over multi-day steps, so it
            # is the diurnal mean over the Time axis, not the instantaneous value.
            solar = _read_var_diurnal_mean(
                nc, "fluxsurf_dn_sw", lat_i, lon_i,
                MARS_SOLAR_CONSTANT_W_M2 * (1.0 - min(1.0, tau) * 0.45) / math.pi,
            )
            h2o = _read_var(nc, "col_h2ovapor", (time_i, lat_i, lon_i), 0.0)
            u = _read_var(nc, "u", (time_i, 0, lat_i, lon_i), 0.0)
            v = _read_var(nc, "v", (time_i, 0, lat_i, lon_i), 0.0)
    except Exception:
        return None
    return MarsClimateSample(
        pressure_pa=max(0.0, ps),
        mean_temperature_c=tsurf_k - 273.15,
        dust_opacity=max(0.0, min(5.0, tau)),
        solar_flux_w_m2=max(0.0, solar),
        relative_humidity=max(0.0, min(1.5, h2o / 0.08)),
        wind_speed_m_s=max(0.0, math.hypot(u, v)),
        source="mcd_netcdf",
        scenario=scenario,
        data_path=str(nc_path),
        data_available=True,
        sol=sol % MARS_YEAR_SOLS,
        ls_deg=_ls_deg(sol),
    )


def _read_var(nc, name: str, index: tuple[int, ...], default: float) -> float:
    var = nc.variables.get(name)
    if var is None:
        return default
    try:
        return float(np.array(var.data[index]).squeeze())
    except Exception:
        return default


def _read_var_diurnal_mean(nc, name: str, lat_i: int, lon_i: int, default: float) -> float:
    var = nc.variables.get(name)
    if var is None:
        return default
    try:
        series = np.array(var.data[:, lat_i, lon_i], dtype=float).squeeze()
        if series.size == 0:
            return default
        return float(np.clip(series, 0.0, None).mean())
    except Exception:
        return default


def _resolve_monthly_netcdf(root: Path, prefix: str, month: int) -> Path | None:
    patterns = [
        f"{prefix}_{month:02d}_me.nc",
        f"{prefix}_{month:02d}_sd.nc",
        f"{prefix}_{month:02d}_thermo_ave_me.nc",
        f"{prefix}_{month:02d}_thermo_ave_sd.nc",
        f"{prefix}_{month:02d}_thermo_min_me.nc",
        f"{prefix}_{month:02d}_thermo_min_sd.nc",
        f"{prefix}_{month:02d}_thermo_max_me.nc",
        f"{prefix}_{month:02d}_thermo_max_sd.nc",
    ]
    for name in patterns:
        candidate = root / name
        if candidate.exists():
            return candidate

    month_pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{2}}).+\.nc$", re.IGNORECASE)
    dated_candidates: list[tuple[int, Path]] = []
    for candidate in root.glob(f"{prefix}_*.nc"):
        match = month_pattern.match(candidate.name)
        if match:
            dated_candidates.append((int(match.group(1)), candidate))
    if not dated_candidates:
        return None
    dated_candidates.sort(key=lambda item: (_circular_month_distance(month, item[0]), item[0], item[1].name))
    return dated_candidates[0][1]


def _circular_month_distance(left: int, right: int) -> int:
    delta = abs(left - right)
    return min(delta, 12 - delta)


def _load_fmcd_module(local_path: Path | None):
    candidates: list[Path] = []
    if local_path and local_path.exists():
        base = local_path if local_path.is_dir() else local_path.parent
        candidates.extend(
            [
                base / "MCD_6.1" / "mcd" / "interfaces" / "python",
                base / "mcd" / "interfaces" / "python",
                base / "interfaces" / "python",
            ]
        )
    ucrt_bin = Path("C:/tools/msys64/ucrt64/bin")
    if ucrt_bin.exists():
        # Force deterministic DLL resolution order on Windows to avoid mixing
        # incompatible MinGW runtimes from different installations.
        os.environ["PATH"] = str(ucrt_bin.resolve()) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(ucrt_bin.resolve()))
        except Exception:
            pass
    for candidate in candidates:
        if candidate.exists():
            path_str = str(candidate.resolve())
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            try:
                os.add_dll_directory(path_str)
            except Exception:
                pass
    try:
        return importlib.import_module("fmcd")
    except Exception:
        return None


def _resolve_mcd_dataset_path(local_path: Path | None, scenario: str) -> Path | None:
    if not local_path:
        return None
    base = local_path if local_path.is_dir() else local_path.parent
    candidates = [
        base / "MCD_6.1" / "data",
        base / "data",
    ]
    local_scenario_dir = base / _scenario_folder(scenario)
    for candidate in candidates:
        if not (candidate.exists() and candidate.is_dir()):
            continue
        scenario_dir = candidate / _scenario_folder(scenario)
        if not scenario_dir.exists() and local_scenario_dir.exists() and local_scenario_dir.is_dir():
            try:
                os.symlink(local_scenario_dir.resolve(), scenario_dir, target_is_directory=True)
            except Exception:
                pass
        if scenario_dir.exists() or scenario.lower() == "climatology":
            return candidate
    return None


def _scenario_to_dust_code(scenario: str) -> int:
    scenario = scenario.lower()
    if scenario == "clim_mineuv":
        return 2
    if scenario == "clim_maxeuv":
        return 3
    if scenario in {"strm", "dust_storm", "storm"}:
        return 5
    if scenario == "warm":
        return 7
    if scenario == "cold":
        return 8
    return 1


def _sample_call_mcd(
    fmcd_module,
    dataset_path: Path,
    scenario: str,
    sol: float,
    latitude_deg: float,
    longitude_deg: float,
    local_hour: float,
    elevation_m: float,
) -> MarsClimateSample | None:
    try:
        mcd = getattr(fmcd_module, "mcd")
        dust_code = _scenario_to_dust_code(scenario)
        extvarkeys = np.ones(100)
        ls_deg = _ls_deg(sol)
        dset_arg = str(dataset_path)
        if not dset_arg.endswith(("/", "\\")):
            dset_arg = dset_arg + os.sep
        (
            pres,
            _dens,
            temp_k,
            zonwind,
            merwind,
            _meanvar,
            _extvar,
            _seedout,
            ierr,
        ) = mcd.call_mcd(
            2,  # height above areoid
            float(elevation_m),
            float(longitude_deg % 360.0),
            float(latitude_deg),
            1,  # high-res interpolation
            1,  # Mars date (Ls)
            float(ls_deg),
            float(local_hour % 24.0),
            dset_arg,
            int(dust_code),
            1,  # no perturbation
            0,
            0.0,
            extvarkeys,
        )
        if int(ierr) != 0:
            return None
        dust_proxy = _scenario_dust(scenario, 2.0 * math.pi * (sol % MARS_YEAR_SOLS) / MARS_YEAR_SOLS)
        solar = MARS_SOLAR_CONSTANT_W_M2 * (1.0 - 0.45 * min(1.0, dust_proxy))
        wind = max(0.0, math.hypot(float(zonwind), float(merwind)))
        return MarsClimateSample(
            pressure_pa=max(0.0, float(pres)),
            mean_temperature_c=float(temp_k) - 273.15,
            dust_opacity=max(0.0, min(5.0, dust_proxy)),
            solar_flux_w_m2=max(0.0, solar),
            relative_humidity=0.0,
            wind_speed_m_s=wind,
            source="mcd_call_mcd",
            scenario=scenario,
            data_path=str(dataset_path),
            data_available=True,
            sol=sol % MARS_YEAR_SOLS,
            ls_deg=ls_deg,
        )
    except Exception:
        return None
