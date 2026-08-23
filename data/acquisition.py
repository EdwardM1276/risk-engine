"""Observed-data acquisition, normalization, provenance, and research fallback.

Public mode uses live provider endpoints and reports unavailable fields. It
does not infer numerical credit behavior from documentation or silently create
synthetic observations. Synthetic mode remains available only as an explicit
research option.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from config.params import PORTFOLIO_SEGMENTS

PROVINCES: List[str] = [
    "Gauteng", "KZN", "Western Cape", "Eastern Cape", "Free State",
    "Mpumalanga", "Limpopo", "North West", "Northern Cape",
]
PROVINCE_WEIGHTS: np.ndarray = np.array(
    [0.35, 0.20, 0.15, 0.08, 0.06, 0.06, 0.05, 0.03, 0.02]
)


@dataclass
class RawDataset:
    """Container for all raw dataframes with audit metadata."""

    macro: pd.DataFrame
    loadshedding: pd.DataFrame
    markets: pd.DataFrame
    timeseries: pd.DataFrame
    portfolio: pd.DataFrame
    acquisition_date: datetime = field(default_factory=datetime.now)
    data_quality: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        """Return a mapping-compatible representation."""
        return {
            "macro": self.macro,
            "loadshedding": self.loadshedding,
            "markets": self.markets,
            "timeseries": self.timeseries,
            "portfolio": self.portfolio,
            "acquisition_date": self.acquisition_date,
            "data_quality": self.data_quality,
        }

    def __getitem__(self, key: str):
        return self.as_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()

    def get(self, key: str, default=None):
        return self.as_dict().get(key, default)

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()

    def values(self):
        return self.as_dict().values()

    def __iter__(self):
        return iter(self.as_dict())


@dataclass
class AcquisitionConfig:
    """Controls caching, raw landing, and configurable provider endpoints."""

    raw_dir: Path = field(default_factory=lambda: Path("data") / "raw")
    cache_dir: Path = field(default_factory=lambda: Path("data") / "cache")
    timeout_seconds: int = 30
    land_raw: bool = True
    cache_enabled: bool = False
    user_agent: str = "risk-engine-public-data/1.0"
    provider_urls: Dict[str, str] = field(default_factory=dict)
    credential_environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class RawArtifact:
    """Audit record for one landed provider response or normalized extract."""

    provider: str
    source_url: str
    retrieved_at: str
    path: str
    sha256: str
    media_type: str

    def as_dict(self) -> Dict[str, str]:
        return self.__dict__.copy()


def fetch_binary_url(
    url: str,
    provider: str,
    config: Optional[AcquisitionConfig] = None,
    suffix: str = ".bin",
) -> bytes:
    """Download a binary public or licensed artifact and land its exact bytes."""
    active_config = config or AcquisitionConfig()
    try:
        response = requests.get(
            url,
            timeout=active_config.timeout_seconds,
            headers={"User-Agent": active_config.user_agent},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Binary provider request failed: {url}: {exc}") from exc
    if active_config.land_raw:
        land_raw_artifact(response.content, provider, url, active_config, suffix)
    return response.content


def land_raw_artifact(
    content: bytes,
    provider: str,
    source_url: str,
    config: AcquisitionConfig,
    suffix: str = ".bin",
) -> RawArtifact:
    """Write immutable content and a checksum manifest to the raw landing zone."""
    digest = hashlib.sha256(content).hexdigest()
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{provider}_{digest[:12]}{suffix}"
    path = config.raw_dir / filename
    path.write_bytes(content)
    artifact = RawArtifact(
        provider=provider,
        source_url=source_url,
        retrieved_at=datetime.now().isoformat(timespec="seconds"),
        path=str(path),
        sha256=digest,
        media_type="application/json" if suffix == ".json" else "text/csv",
    )
    manifest = config.raw_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(artifact.as_dict()) + "\n")
    return artifact


def fetch_csv_url(
    url: str,
    provider: str,
    config: Optional[AcquisitionConfig] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Fetch a configurable CSV provider and optionally land its raw response."""
    active_config = config or AcquisitionConfig()
    cache_path = active_config.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:20]}.csv"
    if active_config.cache_enabled and cache_path.exists():
        return pd.read_csv(cache_path, **read_csv_kwargs)
    try:
        response = requests.get(
            url,
            timeout=active_config.timeout_seconds,
            headers={"User-Agent": active_config.user_agent},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"CSV provider request failed: {url}: {exc}") from exc
    if active_config.land_raw:
        land_raw_artifact(response.content, provider, url, active_config, ".csv")
    if active_config.cache_enabled:
        active_config.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
    try:
        return pd.read_csv(StringIO(response.text), **read_csv_kwargs)
    except (pd.errors.ParserError, ValueError) as exc:
        raise RuntimeError(f"CSV provider returned an invalid payload: {url}") from exc


def fetch_configured_source(
    url: str,
    provider: str,
    config: Optional[AcquisitionConfig] = None,
) -> pd.DataFrame:
    """Fetch a configured official or licensed CSV source.

    This adapter is intentionally format-neutral so SARB, IMF, JSE, Eskom,
    and EBA exports can be plugged in without embedding credentials or URLs.
    """
    return fetch_csv_url(url, provider, config)


def fetch_freddie_mac_artifact(
    url: str,
    config: Optional[AcquisitionConfig] = None,
) -> bytes:
    """Download a Freddie Mac loan-level file after the user's approved login."""
    return fetch_binary_url(url, "freddie_mac_sflld", config, ".zip")


def fetch_uci_credit_card_defaults(
    config: Optional[AcquisitionConfig] = None,
) -> pd.DataFrame:
    """Fetch the documented UCI Taiwan credit-card default observations.

    This is a real observed research dataset with 30,000 records, six months
    of payment-status history, credit limits, bills, payments, and a default
    outcome. It is a calibration benchmark, not a substitute for bank data.
    """
    url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    content = fetch_binary_url(url, "uci_credit_card_defaults", config, ".zip")
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            workbook = next(
                name for name in archive.namelist()
                if name.lower().endswith((".xls", ".xlsx"))
            )
            frame = pd.read_excel(BytesIO(archive.read(workbook)), header=1)
    except (zipfile.BadZipFile, StopIteration, ValueError) as exc:
        raise RuntimeError("UCI credit-default download could not be parsed") from exc
    frame = frame.rename(columns={
        "default payment next month": "default_flag",
        "LIMIT_BAL": "credit_limit",
    })
    payment_columns = [f"PAY_{month}" for month in [0, 2, 3, 4, 5, 6]]
    missing = [column for column in ["ID", "credit_limit", "default_flag", *payment_columns] if column not in frame]
    if missing:
        raise RuntimeError(f"UCI credit-default dataset missing columns: {missing}")
    result = frame.rename(columns={"ID": "account_id"}).copy()
    result["max_dpd_bucket"] = result[payment_columns].max(axis=1).clip(lower=0)
    result["default_flag"] = result["default_flag"].astype(bool)
    result["utilisation_proxy"] = (
        result[[f"BILL_AMT{month}" for month in range(1, 7)]].max(axis=1)
        / result["credit_limit"].replace(0, np.nan)
    )
    return result


def summarize_uci_default_benchmark(frame: pd.DataFrame) -> Dict[str, object]:
    """Summarize observed UCI default rates for calibration and challenger tests."""
    required = {"default_flag", "credit_limit", "max_dpd_bucket", "utilisation_proxy"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"UCI benchmark missing columns: {missing}")
    clean = frame.dropna(subset=["default_flag", "credit_limit", "max_dpd_bucket"])
    if clean.empty:
        raise ValueError("UCI benchmark has no usable observations")
    by_dpd = (
        clean.assign(dpd_band=pd.cut(
            clean["max_dpd_bucket"], bins=[-1, 0, 1, 2, 8, np.inf],
            labels=["current", "one_month", "two_months", "three_to_eight", "nine_plus"],
        ))
        .groupby("dpd_band", observed=False)["default_flag"]
        .agg(default_rate="mean", observations="size")
        .reset_index()
    )
    return {
        "dataset": "UCI Default of Credit Card Clients",
        "observations": int(len(clean)),
        "overall_default_rate": float(clean["default_flag"].mean()),
        "default_rate_by_dpd": by_dpd.to_dict("records"),
        "median_credit_limit": float(clean["credit_limit"].median()),
        "observed_fields": ["credit_limit", "payment_status_history", "bill_amounts", "payments", "default_flag"],
        "missing_model_fields": ["collateral", "recoveries", "IFRS9_stage", "internal_rating", "monthly_observation_date"],
    }


def fetch_sec_company_facts(
    cik: str,
    config: Optional[AcquisitionConfig] = None,
) -> Dict[str, object]:
    """Fetch public SEC XBRL facts for a bank or holding company."""
    active_config = config or AcquisitionConfig()
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    try:
        response = requests.get(
            url,
            timeout=active_config.timeout_seconds,
            headers={"User-Agent": active_config.user_agent},
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"SEC XBRL request failed: {url}: {exc}") from exc
    if active_config.land_raw:
        land_raw_artifact(response.content, "sec_xbrl", url, active_config, ".json")
    return payload


def fetch_fred_series(
    series_ids: List[str],
    periods: int = 36,
    end_date: Optional[pd.Timestamp] = None,
    config: Optional[AcquisitionConfig] = None,
) -> pd.DataFrame:
    """Fetch multiple FRED CSV series and return a month-end observation panel."""
    active_config = config or AcquisitionConfig()
    query = ",".join(series_ids)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={query}"
    frame = fetch_csv_url(url, "fred", active_config, na_values=["."])
    date_column = "observation_date"
    if date_column not in frame.columns:
        raise RuntimeError("FRED response did not contain observation_date")
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.set_index(date_column).apply(pd.to_numeric, errors="coerce")
    monthly = frame.resample("ME").last().dropna(how="all")
    if end_date is not None:
        monthly = monthly[monthly.index <= pd.Timestamp(end_date)]
    monthly = monthly.tail(periods).reset_index().rename(columns={date_column: "date"})
    if monthly.empty:
        raise RuntimeError(f"FRED returned no observations for {series_ids}")
    return monthly


def load_institutional_portfolio(
    path: str,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load an anonymized bank extract from CSV/XLSX with a strict contract."""
    required = required_columns or [
        "account_id", "observation_date", "segment", "principal_outstanding",
        "undrawn_limit", "credit_limit", "utilisation", "collateral_value",
        "province", "months_on_book", "tenure_years", "loadshedding_vulnerability_score",
        "dpd", "internal_rating", "default_flag", "default_date", "recovery_cashflow",
        "recovery_date", "ifrs9_stage", "debt_review_flag", "judgement_flag",
        "administration_order", "base_segment_ttc_pd", "base_segment_lgd",
        "base_segment_ccf", "base_segment_corr",
    ]
    source = Path(path)
    if source.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("Institutional portfolio must be CSV or Excel")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Institutional portfolio missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Institutional portfolio is empty")
    numeric = [
        "principal_outstanding", "undrawn_limit", "credit_limit", "utilisation",
        "collateral_value", "dpd", "recovery_cashflow",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["principal_outstanding"] <= 0).any():
        raise ValueError("Institutional portfolio contains non-positive principal")
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="raise")
    if frame.duplicated(["account_id", "observation_date"]).any():
        raise ValueError("Institutional portfolio contains duplicate account-period keys")
    return frame


def validate_provider_frame(frame: pd.DataFrame, required_columns: List[str], name: str) -> List[str]:
    """Return auditable schema and missingness flags for any provider frame."""
    flags = [f"{name}: missing column {column}" for column in required_columns if column not in frame]
    if not frame.empty:
        flags.extend(
            f"{name}: all values missing in {column}"
            for column in required_columns
            if column in frame and frame[column].isna().all()
        )
    else:
        flags.append(f"{name}: empty dataframe")
    return flags


def _ensure_positive(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Lightweight data quality validator -- returns list of failing flags."""
    flags: List[str] = []
    for c in cols:
        if c in df.columns and (df[c] <= 0).any():
            flags.append(f"Non-positive in {c}: {(df[c] <= 0).sum()} rows")
    return flags


def _get_json(url: str, params: Optional[Dict[str, object]] = None) -> object:
    """Get a public JSON endpoint with a bounded timeout and useful errors."""
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Public data request failed: {url}: {exc}") from exc


def fetch_sarb_current_rates(
    config: Optional[AcquisitionConfig] = None,
) -> Dict[str, float]:
    """Fetch observed current repo and prime rates from SARB's WebIndicators API."""
    url = "https://custom.resbank.co.za/SarbWebApi/WebIndicators/OtherIndicators/HistoricalDatesOfRateChanges/"
    payload = _get_json(url)
    if not isinstance(payload, list):
        raise RuntimeError("SARB rate endpoint returned an invalid payload")
    rates: Dict[str, float] = {}
    for row in payload:
        code = row.get("TsCode")
        if code == "MRDREPOR":
            rates["repo_rate"] = float(row["TheValue"]) / 100.0
        elif code == "MRDPRIME":
            rates["prime_rate"] = float(row["TheValue"]) / 100.0
    if set(rates) != {"repo_rate", "prime_rate"}:
        raise RuntimeError("SARB rate endpoint did not return repo and prime rates")
    return rates


def fetch_world_bank_macro_data(
    country: str = "ZAF",
    periods: int = 36,
) -> pd.DataFrame:
    """Fetch observed annual macro indicators from the World Bank API.

    Annual observations are expanded to month-end rows because the existing
    stress and reporting layers use a monthly time-series contract. Expansion
    is labelled as interpolation; it is not additional observed information.
    """
    indicators = {
        "NY.GDP.MKTP.KD.ZG": "gdp_yoy",
        "FP.CPI.TOTL.ZG": "cpi_yoy",
        "SL.UEM.TOTL.ZS": "unemployment_rate",
        "PA.NUS.FCRF": "zar_usd",
    }
    series: Dict[str, pd.Series] = {}
    for indicator, name in indicators.items():
        payload = _get_json(
            f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}",
            {"format": "json", "per_page": 100},
        )
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            raise RuntimeError(f"World Bank returned no observations for {indicator}")
        values = {
            pd.Timestamp(f"{row['date']}-12-31"): float(row["value"])
            for row in payload[1]
            if row.get("value") is not None
        }
        series[name] = pd.Series(values).sort_index()

    annual = pd.concat(series, axis=1).dropna().sort_index()
    if annual.empty:
        raise RuntimeError(f"World Bank returned no complete macro observations for {country}")
    monthly_index = pd.date_range(end=annual.index.max(), periods=periods, freq="ME").normalize()
    monthly = annual.reindex(annual.index.union(monthly_index)).sort_index().interpolate("time")
    monthly = monthly.reindex(monthly_index).ffill().bfill().reset_index()
    monthly = monthly.rename(columns={"index": "date"})
    # World Bank rates and growth values are percentages; model contracts use fractions.
    for column in ["cpi_yoy", "gdp_yoy", "unemployment_rate"]:
        monthly[column] = monthly[column] / 100.0
    monthly["repo_rate"] = np.nan
    monthly["prime_rate"] = np.nan
    return monthly[["date", "repo_rate", "prime_rate", "cpi_yoy", "gdp_yoy",
                    "unemployment_rate", "zar_usd"]]


def fetch_fdic_bank_financials(
    state: Optional[str] = None,
    banks: int = 25,
    periods: int = 12,
) -> pd.DataFrame:
    """Fetch observed quarterly bank financials from the FDIC BankFind API.

    This is an aggregate bank panel, not account-level performance data. It is
    therefore normalized into bank-observation rows for benchmarking and
    portfolio-scale experiments, while preserving its aggregate provenance.
    """
    filters: List[str] = []
    if state:
        filters.append(f"STALP:{state.upper()}")
    payload = _get_json(
        "https://banks.data.fdic.gov/api/financials",
        {
            "filters": ",".join(filters),
            "limit": max(banks * periods * 3, 100),
            "fields": "NAME,CERT,REPDTE,ASSET,NETINC,DEP,LNLS",
            "sort_by": "REPDTE",
            "sort_order": "DESC",
        },
    )
    records = payload.get("data", []) if isinstance(payload, dict) else []
    rows = []
    for wrapper in records:
        record = wrapper.get("data", wrapper) if isinstance(wrapper, dict) else {}
        assets = record.get("ASSET")
        loans = record.get("LNLS")
        if assets is None or loans is None or float(loans) <= 0:
            continue
        rows.append({
            "date": pd.to_datetime(str(record["REPDTE"]), format="%Y%m%d"),
            "bank_name": record.get("NAME", "Unknown bank"),
            "bank_cert": record.get("CERT"),
            "total_assets_usd": float(assets) * 1000.0,
            "loan_balance_usd": float(loans) * 1000.0,
            "deposits_usd": float(record.get("DEP") or 0.0) * 1000.0,
            "net_income_usd": float(record.get("NETINC") or 0.0) * 1000.0,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("FDIC returned no bank financial observations with loan balances")
    result = result.sort_values("date", ascending=False)
    return result.groupby("bank_name", sort=False).head(periods).head(banks * periods).reset_index(drop=True)


def fetch_fred_exchange_rate(
    periods: int = 36,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Fetch observed South African rand exchange rates from FRED CSV."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSFUS"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text), na_values=["."])
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"FRED market-data request failed: {url}: {exc}") from exc
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["DEXSFUS"] = pd.to_numeric(frame["DEXSFUS"], errors="coerce")
    frame = frame.dropna(subset=["DEXSFUS"]).rename(
        columns={"observation_date": "date", "DEXSFUS": "zar_usd"}
    )
    monthly = frame.set_index("date")["zar_usd"].resample("ME").last().dropna()
    if end_date is not None:
        monthly = monthly[monthly.index <= pd.Timestamp(end_date)]
    monthly = monthly.tail(periods)
    if monthly.empty:
        raise RuntimeError("FRED returned no exchange-rate observations")
    return pd.DataFrame({
        "date": monthly.index,
        "jse_alsi": np.nan,
        "jse_property": np.nan,
        "sovereign_cds_bps": np.nan,
        "gold_price_zar": np.nan,
        "platinum_price_zar": np.nan,
        "coal_price_zar": np.nan,
        "zar_usd": monthly.to_numpy(),
    }).reset_index(drop=True)


def empty_loadshedding_data() -> pd.DataFrame:
    """Return the explicit no-source contract for historical loadshedding data."""
    return pd.DataFrame(columns=["date", "load_shedding_stage", "daily_hours_shed", "monthly_gwh_lost"])


def normalize_fdic_portfolio(
    financials: pd.DataFrame,
    institution_size: str = "Public_US_Bank_Panel",
) -> pd.DataFrame:
    """Map observed FDIC bank loan balances into the engine's input contract."""
    rows = []
    for idx, record in financials.reset_index(drop=True).iterrows():
        principal = float(record["loan_balance_usd"])
        rows.append({
            "account_id": f"FDIC_{record['bank_cert']}_{record['date']:%Y%m%d}",
            "segment": "Corporate_Large",
            "institution_size": institution_size,
            "province": "US",
            "principal_outstanding": principal,
            "undrawn_limit": 0.0,
            "collateral_value": principal,
            "loan_to_value": 1.0,
            "tenure_years": 1,
            "months_on_book": 12,
            "dpd": 0,
            "debt_review_flag": False,
            "judgement_flag": False,
            "administration_order": False,
            "loadshedding_vulnerability_score": 1,
            "internal_rating": "BBB",
            "base_segment_ccf": PORTFOLIO_SEGMENTS["Corporate_Large"]["ccf"],
            "base_segment_ttc_pd": PORTFOLIO_SEGMENTS["Corporate_Large"]["ttc_pd"],
            "base_segment_lgd": PORTFOLIO_SEGMENTS["Corporate_Large"]["lgd"],
            "base_segment_corr": PORTFOLIO_SEGMENTS["Corporate_Large"]["corr"],
            "source_bank": record["bank_name"],
            "source_date": record["date"],
        })
    return pd.DataFrame(rows)


def fetch_sarb_macro_data(periods: int = 36, seed: int = 42) -> pd.DataFrame:
    """Generate SARB-style macro timeseries (MMRD000A prime, CPI1000F CPI, etc)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="ME")
    repo = np.clip(0.0775 + rng.normal(0, 0.003, periods).cumsum(), 0.035, 0.12)
    prime = repo + 0.035
    cpi_yoy = np.clip(0.050 + rng.normal(0, 0.002, periods).cumsum(), 0.02, 0.14)
    gdp_yoy = np.clip(0.012 + rng.normal(0, 0.004, periods).cumsum(), -0.08, 0.05)
    unemployment = np.clip(0.325 + rng.normal(0, 0.003, periods).cumsum(), 0.20, 0.50)
    zar_usd = np.clip(18.5 + rng.normal(0, 0.2, periods).cumsum(), 14.0, 25.0)
    df = pd.DataFrame({
        "date": dates, "repo_rate": repo, "prime_rate": prime,
        "cpi_yoy": cpi_yoy, "gdp_yoy": gdp_yoy,
        "unemployment_rate": unemployment, "zar_usd": zar_usd,
    })
    return df


def fetch_eskom_loadshedding(periods: int = 36, seed: int = 42) -> pd.DataFrame:
    """EskomSePush-style loadshedding stages with winter (May-Aug) spike."""
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="ME")
    base_stage = rng.integers(1, 5, periods)
    winter_bonus = np.where(dates.month.isin([5, 6, 7, 8]), rng.integers(1, 3, periods), 0)
    stage = np.clip(base_stage + winter_bonus, 0, 8)
    hours = stage * (2 + rng.uniform(0, 2, periods))
    gwh_lost = hours * rng.uniform(4.0, 6.0, periods) * 30
    return pd.DataFrame({
        "date": dates, "load_shedding_stage": stage,
        "daily_hours_shed": hours, "monthly_gwh_lost": gwh_lost,
    })


def fetch_jse_market_data(periods: int = 36, seed: int = 42) -> pd.DataFrame:
    """JSE indices + sovereign CDS + rand-denominated commodity prices."""
    rng = np.random.default_rng(seed + 2)
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="ME")
    jse_alsi = 75000 * np.exp(rng.normal(0.006, 0.04, periods).cumsum())
    jse_property = 2500 * np.exp(rng.normal(0.001, 0.05, periods).cumsum())
    sovereign_cds_bps = np.clip(250 + rng.normal(0, 20, periods).cumsum(), 80, 700)
    gold = 1200000 * np.exp(rng.normal(0.003, 0.035, periods).cumsum())
    platinum = 1800000 * np.exp(rng.normal(0.002, 0.05, periods).cumsum())
    coal = 1800 * np.exp(rng.normal(0.002, 0.06, periods).cumsum())
    return pd.DataFrame({
        "date": dates, "jse_alsi": jse_alsi, "jse_property": jse_property,
        "sovereign_cds_bps": sovereign_cds_bps,
        "gold_price_zar": gold, "platinum_price_zar": platinum, "coal_price_zar": coal,
    })


def generate_sa_loan_portfolio(
    total_exposure: float = 500_000_000_000.0,
    n_accounts: int = 5000,
    seed: int = 2024,
    institution_size: str = "Large_D-SIB",
) -> pd.DataFrame:
    """Generate a realistic SA multi-segment loan book.

    Fields include NCA-required dpd buckets, debt-review and judgement flags,
    loadshedding vulnerability score (1-5), 9 provinces, 7 product segments,
    and internal rating (AAA to CCC).
    """
    rng = np.random.default_rng(seed)
    accounts: List[Dict] = []
    for seg, seg_params in PORTFOLIO_SEGMENTS.items():
        n_seg = max(1, int(n_accounts * seg_params["weight"]))
        seg_exposure = total_exposure * seg_params["weight"]
        avg_exposure = seg_exposure / n_seg

        is_retail = seg.startswith("Retail")
        is_sme = "SME" in seg
        is_corp = "Corporate" in seg and not is_sme
        is_sov = "Sovereign" in seg

        for _ in range(n_seg):
            principal = avg_exposure * rng.lognormal(0, 0.8)
            if is_retail and "Mortgage" in seg:
                undrawn = principal * rng.uniform(0.05, 0.30)
                collateral = principal * rng.uniform(1.1, 1.6)
                province = rng.choice(PROVINCES, p=PROVINCE_WEIGHTS)
                ls_vuln = int(rng.choice([1, 2, 3, 4, 5],
                                         p=[0.10, 0.25, 0.35, 0.20, 0.10]))
                tenure = int(rng.integers(5, 31))
            elif is_retail and "Vehicle" in seg:
                undrawn = principal * rng.uniform(0.0, 0.10)
                collateral = principal * rng.uniform(0.6, 1.1)
                province = rng.choice(PROVINCES, p=PROVINCE_WEIGHTS)
                ls_vuln = int(rng.choice([1, 2, 3, 4, 5],
                                         p=[0.15, 0.30, 0.30, 0.15, 0.10]))
                tenure = int(rng.integers(1, 8))
            elif is_retail and "CreditCard" in seg:
                undrawn = principal * rng.uniform(0.3, 1.0)
                collateral = 0.0
                province = rng.choice(PROVINCES, p=PROVINCE_WEIGHTS)
                ls_vuln = int(rng.choice([1, 2, 3, 4, 5],
                                         p=[0.15, 0.30, 0.30, 0.15, 0.10]))
                tenure = 0
            elif is_retail and "Overdraft" in seg:
                undrawn = principal * rng.uniform(0.2, 1.2)
                collateral = 0.0
                province = rng.choice(PROVINCES, p=PROVINCE_WEIGHTS)
                ls_vuln = int(rng.choice([1, 2, 3, 4, 5],
                                         p=[0.15, 0.25, 0.30, 0.20, 0.10]))
                tenure = 0
            elif is_sme:
                undrawn = principal * rng.uniform(0.1, 0.5)
                collateral = principal * rng.uniform(0.4, 1.2)
                province = rng.choice(PROVINCES, p=PROVINCE_WEIGHTS)
                ls_vuln = int(rng.choice([1, 2, 3, 4, 5],
                                         p=[0.05, 0.15, 0.30, 0.35, 0.15]))
                tenure = int(rng.integers(1, 11))
            elif is_corp:
                undrawn = principal * rng.uniform(0.05, 0.4)
                collateral = principal * rng.uniform(0.5, 1.3)
                province = rng.choice(["Gauteng", "Western Cape", "KZN"],
                                      p=[0.6, 0.25, 0.15])
                ls_vuln = int(rng.choice([1, 2, 3, 4, 5],
                                         p=[0.05, 0.15, 0.35, 0.30, 0.15]))
                tenure = int(rng.integers(2, 16))
            else:  # Sovereign / Bank segment
                undrawn = principal * rng.uniform(0.0, 0.2)
                collateral = principal * rng.uniform(0.8, 1.5)
                province = "Gauteng"
                ls_vuln = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
                tenure = int(rng.integers(3, 21))

            dpd = int(rng.choice([0, 0, 0, 0, 5, 15, 30, 60, 90, 120],
                                 p=[0.55, 0.15, 0.08, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.01]))
            debt_review = bool(dpd >= 60 and rng.random() < 0.40)
            judgement = bool(dpd >= 90 and rng.random() < 0.30)
            admin_order = bool(dpd >= 120)
            internal_rating = str(rng.choice(
                ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"],
                p=[0.03, 0.08, 0.18, 0.28, 0.22, 0.14, 0.07],
            ))

            accounts.append({
                "segment": seg,
                "institution_size": institution_size,
                "province": province,
                "principal_outstanding": float(principal),
                "undrawn_limit": float(undrawn),
                "collateral_value": float(collateral),
                "loan_to_value": float(principal / collateral) if collateral > 0 else np.inf,
                "tenure_years": tenure,
                "months_on_book": int(rng.integers(1, 241)),
                "dpd": dpd,
                "debt_review_flag": debt_review,
                "judgement_flag": judgement,
                "administration_order": admin_order,
                "loadshedding_vulnerability_score": ls_vuln,
                "internal_rating": internal_rating,
                "base_segment_ccf": float(seg_params["ccf"]),
                "base_segment_ttc_pd": float(seg_params["ttc_pd"]),
                "base_segment_lgd": float(seg_params["lgd"]),
                "base_segment_corr": float(seg_params["corr"]),
            })

    df = pd.DataFrame(accounts)
    df.insert(0, "account_id", [f"ACC_{i + 1:06d}" for i in range(len(df))])
    return df


def acquire_all_data(
    total_exposure: float = 500_000_000_000.0,
    n_accounts: int = 5000,
    periods: int = 36,
    seed: int = 2024,
    institution_size: str = "Large_D-SIB",
    data_source: str = "synthetic",
    allow_synthetic_fallback: bool = False,
    portfolio_path: Optional[str] = None,
    acquisition_config: Optional[AcquisitionConfig] = None,
) -> RawDataset:
    """Acquire a validated bundle from synthetic or public sources.

    ``public`` uses World Bank, FRED, and FDIC data. It does not silently
    invent account performance or loadshedding history when those sources are
    absent. ``allow_synthetic_fallback`` exists for demos and is recorded.
    """
    if data_source not in {"synthetic", "public", "institutional"}:
        raise ValueError("data_source must be 'synthetic', 'public', or 'institutional'")
    if data_source == "institutional" and not portfolio_path:
        raise ValueError("institutional mode requires --portfolio-path or portfolio_path")
    active_config = acquisition_config or AcquisitionConfig()
    source_notes: Dict[str, str]
    if data_source in {"public", "institutional"}:
        try:
            macro = fetch_world_bank_macro_data(periods=periods)
            current_rates = fetch_sarb_current_rates(active_config)
            if portfolio_path:
                portfolio = load_institutional_portfolio(portfolio_path)
                portfolio_source = f"Institutional extract: {portfolio_path}"
            else:
                financials = fetch_fdic_bank_financials(banks=max(1, min(n_accounts, 25)), periods=periods)
                portfolio = normalize_fdic_portfolio(financials, institution_size)
                portfolio_source = "FDIC BankFind quarterly aggregate bank financials; not account-level data"
            mkts = fetch_fred_exchange_rate(periods, end_date=macro["date"].max())
            ls = empty_loadshedding_data()
            merged_ts = macro.merge(mkts, on="date", how="inner", suffixes=("", "_market"))
            source_notes = {
                "macro": "World Bank API; annual observations interpolated to month-end",
                "current_rates": "SARB WebIndicators current repo and prime rate snapshot",
                "portfolio": portfolio_source,
                "markets": "Federal Reserve Bank of St. Louis FRED DEXSFUS daily observations resampled monthly",
                "loadshedding": "No public source configured; dataframe intentionally empty",
            }
        except RuntimeError:
            if not allow_synthetic_fallback:
                raise
            macro = fetch_sarb_macro_data(periods, seed)
            ls = fetch_eskom_loadshedding(periods, seed)
            mkts = fetch_jse_market_data(periods, seed)
            portfolio = generate_sa_loan_portfolio(total_exposure, n_accounts, seed, institution_size)
            merged_ts = macro.merge(ls, on="date", how="inner").merge(mkts, on="date", how="inner")
            source_notes = {"all": "Synthetic fallback used after public acquisition failure"}
    else:
        macro = fetch_sarb_macro_data(periods, seed)
        ls = fetch_eskom_loadshedding(periods, seed)
        mkts = fetch_jse_market_data(periods, seed)
        portfolio = generate_sa_loan_portfolio(total_exposure, n_accounts, seed, institution_size)
        merged_ts = macro.merge(ls, on="date", how="inner").merge(mkts, on="date", how="inner")
        source_notes = {"all": "Synthetic research generator"}

    quality_flags: List[str] = []
    quality_flags.extend(_ensure_positive(macro, ["zar_usd"]))
    quality_flags.extend(_ensure_positive(portfolio, ["principal_outstanding"]))
    for column in macro.columns:
        if column != "date" and macro[column].isna().any():
            quality_flags.append(f"Missing macro observations in {column}")
    for column in mkts.columns:
        if column != "date" and mkts[column].isna().all():
            quality_flags.append(f"No market observations in {column}")
    if portfolio["loan_to_value"].isna().any():
        quality_flags.append("NaN LTV rows present")

    placeholder_fields = (
        ["dpd", "debt_review_flag", "judgement_flag", "administration_order",
         "internal_rating", "undrawn_limit", "collateral_value"]
        if data_source in {"public", "institutional"} and not source_notes.get("all") and not portfolio_path else []
    )
    validation_ready = bool(
        data_source == "institutional"
        and not placeholder_fields
        and not macro[["cpi_yoy", "gdp_yoy", "unemployment_rate", "zar_usd"]].isna().any().any()
        and not portfolio.empty
    )

    artifacts = []
    if active_config.land_raw:
        for name, frame in [("macro", macro), ("loadshedding", ls), ("markets", mkts), ("portfolio", portfolio)]:
            artifact = land_raw_artifact(
                frame.to_csv(index=False).encode("utf-8"),
                f"normalized_{name}",
                source_notes.get(name, source_notes.get("all", "synthetic")),
                active_config,
                ".csv",
            )
            artifacts.append(artifact.as_dict())

    return RawDataset(
        macro=macro, loadshedding=ls, markets=mkts, timeseries=merged_ts,
        portfolio=portfolio,
        data_quality={
            "n_records": int(len(merged_ts)),
            "n_accounts": int(len(portfolio)),
            "total_exposure_requested": float(total_exposure),
            "gap_flags": quality_flags,
            "placeholder_fields": placeholder_fields,
            "validation_ready": validation_ready,
            "raw_artifacts": artifacts,
            "current_rates": current_rates if data_source in {"public", "institutional"} and not source_notes.get("all") else {},
            "data_source": data_source,
            "source_notes": source_notes,
            "synthetic_fallback_used": bool(data_source == "public" and source_notes.get("all")),
            "status": "PASSED" if len(quality_flags) == 0 else "REVIEW",
        },
    )
