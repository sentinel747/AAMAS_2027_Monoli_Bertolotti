from types import SimpleNamespace

import src.data.mars_climate as mars_climate
from src.data.mars_climate import MarsClimateProvider
from src.simulation.planetary_coupling import PlanetaryCoupler
from src.world.world_generator import WorldGenerator


def _install_fmcd_mock(monkeypatch, *, ierr: int = 0):
    class _McdMock:
        @staticmethod
        def call_mcd(*_args):
            return (640.0, 0.02, 215.0, 3.0, 4.0, [], [], 0, ierr)

    fmcd_mock = SimpleNamespace(mcd=_McdMock())
    monkeypatch.setattr(mars_climate.importlib, "import_module", lambda name: fmcd_mock if name == "fmcd" else None)


def test_mars_climate_provider_uses_call_mcd_provider(tmp_path, monkeypatch):
    dataset = tmp_path / "MCD_6.1" / "data"
    (dataset / "warm").mkdir(parents=True)
    _install_fmcd_mock(monkeypatch)

    provider = MarsClimateProvider(
        {
            "climate": {
                "provider": "call_mcd",
                "data_path": str(tmp_path),
                "scenario": "warm",
                "latitude_deg": -4.6,
                "longitude_deg": 137.4,
            }
        }
    )

    sample = provider.sample(0.0)

    assert sample.source == "mcd_call_mcd"
    assert sample.data_path == str(dataset)
    assert sample.pressure_pa == 640.0
    assert sample.mean_temperature_c == 215.0 - 273.15
    assert sample.wind_speed_m_s == 5.0
    assert provider.metadata()["call_mcd_available"] is True


def test_call_mcd_provider_reports_unavailable_without_fallback(tmp_path, monkeypatch):
    dataset = tmp_path / "MCD_6.1" / "data"
    (dataset / "warm").mkdir(parents=True)
    _install_fmcd_mock(monkeypatch, ierr=14)

    provider = MarsClimateProvider({"climate": {"provider": "call_mcd", "data_path": str(tmp_path), "scenario": "warm"}})
    sample = provider.sample(0.0)

    assert sample.source == "mcd_call_unavailable_proxy"
    assert sample.data_available is False
    assert sample.data_path == str(dataset)


def test_planetary_coupler_requires_call_mcd_and_updates_grid(tmp_path, monkeypatch):
    dataset = tmp_path / "MCD_6.1" / "data"
    (dataset / "warm").mkdir(parents=True)
    _install_fmcd_mock(monkeypatch)

    world = WorldGenerator(seed=22, map_profile="balanced").generate(12, 12)
    before_dust = world.get_cell(6, 6).dust_level
    coupler = PlanetaryCoupler(
        {
            "climate": {"enabled": True, "provider": "call_mcd", "scenario": "warm", "data_path": str(tmp_path), "mcd_path": str(tmp_path)},
            "simulation": {"days_per_step": 365, "max_days": 3650},
        }
    )

    coupler.sync_world(world)
    metrics = world.planetary_state

    assert metrics["climate_source"] == "mcd_call_mcd"
    assert metrics["climate_dust_opacity"] > 0
    assert metrics["solar_flux_w_m2"] >= 0
    assert "pressure_pa" in metrics
    assert world.get_cell(6, 6).dust_level != before_dust
