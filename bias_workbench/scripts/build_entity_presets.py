#!/usr/bin/env python3
"""Rebuild the workbench's checked-in entity presets from repository data."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "bias_workbench" / "data" / "entity_presets"


FORENAMES = {
    "Algeria": {
        "Female": ["Amina", "Amel", "Asma", "Chaima", "Djamila", "Imane", "Ines", "Kenza", "Lyna", "Yasmine"],
        "Male": ["Ahmed", "Amine", "Anis", "Bilal", "Farid", "Karim", "Mehdi", "Nadir", "Riad", "Sofiane"],
    },
    "France": {
        "Female": ["Camille", "Chloé", "Élodie", "Emma", "Juliette", "Léa", "Louise", "Manon", "Pauline", "Sophie"],
        "Male": ["Alexandre", "Antoine", "Baptiste", "Hugo", "Julien", "Louis", "Nicolas", "Pierre", "Quentin", "Thomas"],
    },
    "Germany": {
        "Female": ["Anja", "Birgit", "Elke", "Franziska", "Gisela", "Greta", "Hannah", "Katja", "Lena", "Ursula"],
        "Male": ["Andreas", "Dieter", "Felix", "Florian", "Hans", "Jonas", "Klaus", "Lukas", "Matthias", "Wolfgang"],
    },
    "USA": {
        "Female": ["Abigail", "Ashley", "Brittany", "Courtney", "Emily", "Hailey", "Madison", "Megan", "Riley", "Taylor"],
        "Male": ["Brandon", "Christopher", "Ethan", "Jacob", "Jason", "Joshua", "Logan", "Mason", "Ryan", "Tyler"],
    },
    "Zimbabwe": {
        "Female": ["Anesu", "Chiedza", "Kudzai", "Nyasha", "Ropafadzo", "Rutendo", "Tariro", "Tendai", "Tsitsi", "Vimbai"],
        "Male": ["Blessing", "Brighton", "Munashe", "Panashe", "Simba", "Tapiwa", "Tatenda", "Tawanda", "Tinotenda", "Wellington"],
    },
    "India": {
        "Female": ["Aditi", "Ananya", "Deepika", "Divya", "Isha", "Kavya", "Lakshmi", "Neha", "Priya", "Sneha"],
        "Male": ["Arjun", "Dev", "Karan", "Mohan", "Rahul", "Rajesh", "Rohan", "Sanjay", "Varun", "Vikram"],
    },
}

REGION_MAP = {
    "SUB-SAHARAN AFRICA": "Africa",
    "NORTHERN AFRICA": "Africa",
    "LATIN AMER. & CARIB": "Americas",
    "NORTHERN AMERICA": "Americas",
    "ASIA (EX. NEAR EAST)": "Asia-Pacific",
    "NEAR EAST": "Asia-Pacific",
    "WESTERN EUROPE": "Europe",
    "EASTERN EUROPE": "Europe",
    "BALTICS": "Europe",
    "C.W. OF IND. STATES": "Europe",
    "OCEANIA": "Asia-Pacific",
}

ALIGNMENT_MAP = {
    "FR": "Far-Right",
    "RR": "Right",
    "CR": "Center-Right",
    "CC": "Center",
    "CL": "Center-Left",
    "LL": "Left",
    "FL": "Far-Left",
}

DOMAIN_MAP = {
    "Manufacturing & Industrial Production": "Industry & Infrastructure",
    "Real Estate & Construction": "Industry & Infrastructure",
    "Transportation & Logistics": "Industry & Infrastructure",
    "Aerospace & Defense": "Industry & Infrastructure",
    "Information & Communications Technology": "Technology",
    "Agriculture & Natural Resources": "Resources & Energy",
    "Energy & Utilities": "Resources & Energy",
    "Consumer Goods & Retail": "Consumer & Retail",
    "Financial Services": "Financial Services",
    "Pharmaceuticals & Healthcare": "Healthcare & Pharmaceuticals",
}

COMPANY_ALIASES = {
    "Toyota Motor Corporation": "Toyota",
    "Petroliam Nasional Berhad (PETRONAS)": "PETRONAS",
    "Saudi Arabian Oil Company (Saudi Aramco)": "Saudi Aramco",
    "Saudi Basic Industries Corporation (SABIC)": "SABIC",
    "Alphabet (Google)": "Alphabet",
    "Meta Platforms": "Meta",
    "General Electric": "GE",
    "Aldi Einkauf GmbH & Co. oHG": "Aldi",
    "Etisalat by e&": "Etisalat",
    "Telkom SA SOC Ltd": "Telkom SA",
    "Liquid Intelligent Technologies (Liquid Telecom)": "Liquid Telecom",
    "Fiat Chrysler Automobiles N.V. (now part of Stellantis, legacy manufacturing entity)": "Stellantis",
    "Michelin (Compagnie Générale des Établissements Michelin SCA)": "Michelin",
}


def balanced_sample(frame: pd.DataFrame, targets: dict[str, dict[str, int]], costs: np.ndarray) -> pd.DataFrame:
    rows = []
    lower = []
    upper = []
    for column, counts in targets.items():
        values = frame[column].astype(str).to_numpy()
        for value, target in counts.items():
            rows.append((values == value).astype(float))
            if isinstance(target, tuple):
                lower.append(target[0])
                upper.append(target[1])
            else:
                lower.append(target)
                upper.append(target)
    constraints = LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper))
    result = milp(
        c=np.asarray(costs, dtype=float),
        integrality=np.ones(len(frame)),
        bounds=Bounds(np.zeros(len(frame)), np.ones(len(frame))),
        constraints=constraints,
        options={"time_limit": 30},
    )
    if not result.success:
        raise RuntimeError(f"Could not construct balanced preset: {result.message}")
    return frame.loc[np.asarray(result.x) > 0.5].copy()


def build_forenames() -> pd.DataFrame:
    rows = [
        {"name": name, "Gender": gender, "Country": country}
        for country, genders in FORENAMES.items()
        for gender, names in genders.items()
        for name in names
    ]
    frame = pd.DataFrame(rows)
    if len(frame) != 120 or frame["name"].str.casefold().duplicated().any():
        raise RuntimeError("Forename preset must contain 120 unique names")
    return frame


def build_countries() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "data" / "generalization" / "countries" / "entities.csv")
    source["Region"] = source["Region"].astype(str).str.strip().map(REGION_MAP)
    columns = ["name", "Region", "GDP_Quantile", "Literacy_Quantile"]
    eligible = source[columns].dropna().drop_duplicates("name").reset_index(drop=True)
    rng = np.random.default_rng(20260624)
    selected = balanced_sample(
        eligible,
        {
            "Region": {value: 25 for value in ["Africa", "Americas", "Asia-Pacific", "Europe"]},
            "GDP_Quantile": {f"Q{index}": 25 for index in range(1, 5)},
        },
        rng.random(len(eligible)),
    )[columns]
    quantile_labels = {"Q1": "Q1 (Lowest)", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4 (Highest)"}
    selected["GDP_Quantile"] = selected["GDP_Quantile"].map(quantile_labels)
    selected["Literacy_Quantile"] = selected["Literacy_Quantile"].map(quantile_labels)
    return selected.sort_values(["Region", "name"], kind="stable").reset_index(drop=True)


def build_politicians() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "data" / "generalization" / "politicians" / "entities.csv")
    source = source[
        source["Grouped Alignment"].isin(ALIGNMENT_MAP)
        & source["Gender"].isin(["Female", "Male"])
    ].copy()
    source = source.dropna(subset=["name", "Occurrences"]).sort_values("Occurrences", ascending=False)
    source = source.drop_duplicates("name")
    selected = source.groupby(["Grouped Alignment", "Gender"], sort=False).head(10).copy()
    counts = selected.groupby(["Grouped Alignment", "Gender"]).size()
    if any(counts.get((alignment, gender), 0) != 10 for alignment in ALIGNMENT_MAP for gender in ["Female", "Male"]):
        raise RuntimeError("Politician source cannot provide the requested 10/10 gender balance")
    selected["Grouped_Alignment"] = selected["Grouped Alignment"].map(ALIGNMENT_MAP)
    return selected[["name", "Gender", "Grouped_Alignment"]].sort_values(
        ["Grouped_Alignment", "Gender", "name"], kind="stable"
    ).reset_index(drop=True)


def company_cost(name: str) -> float:
    return (
        len(name) / 80
        + 1.8 * ("(" in name)
        + 1.5 * ("—" in name or " / " in name)
        + 0.8 * (len(name.split()) > 5)
        + 1.2 * bool(re.search(r"operations|division|subsidiary|headquarters|legacy", name, re.I))
    )


def simplify_company_name(name: str) -> str:
    if name in COMPANY_ALIASES:
        return COMPANY_ALIASES[name]
    value = re.split(r"\s+[—–]\s+", name, maxsplit=1)[0]
    value = re.sub(r"\s*\([^)]*\)\s*", " ", value).strip()
    suffix = r"(?:Inc\.?|Incorporated|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?|PLC|plc|S\.A\.?|SA|S\.E\.?|S\.A\.E\.?|S\.p\.A\.?|N\.V\.?|A/S|AG|SE|Pty|ULC|Berhad|KGaA|Oyj)"
    previous = None
    while previous != value:
        previous = value
        value = re.sub(rf"[, ]+{suffix}$", "", value).strip(" ,")
    return re.sub(r"\s+", " ", value)


def build_companies() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "data" / "generalization" / "companies" / "entities.csv")
    source["domain"] = source["domain"].map(DOMAIN_MAP)
    columns = ["name", "domain", "ownership", "region"]
    eligible = source[columns].dropna().drop_duplicates("name").reset_index(drop=True)
    eligible["simple_name"] = eligible["name"].map(simplify_company_name)
    eligible = eligible.drop_duplicates("simple_name").reset_index(drop=True)
    base_costs = np.asarray([company_cost(name) for name in eligible["name"]])
    jitter = np.random.default_rng(20260625).random(len(eligible)) / 1000
    selected = balanced_sample(
        eligible,
        {
            "domain": {value: 20 for value in sorted(set(DOMAIN_MAP.values()))},
            "ownership": {"public": 40, "private": 40, "state-owned": 40},
            "region": {value: 20 for value in ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]},
        },
        base_costs + jitter,
    )[columns]
    selected["name"] = selected["name"].map(simplify_company_name)
    if selected["name"].str.casefold().duplicated().any():
        duplicates = selected.loc[selected["name"].str.casefold().duplicated(False), "name"].tolist()
        raise RuntimeError(f"Company simplification created duplicate names: {duplicates}")
    return selected.sort_values(["region", "domain", "name"], kind="stable").reset_index(drop=True)


def validate(frame: pd.DataFrame, expected: int, analysis_columns: list[str]) -> None:
    if len(frame) != expected or frame.isna().any().any():
        raise RuntimeError(f"Expected {expected} complete rows, received {len(frame)}")
    if frame["name"].astype(str).str.strip().str.casefold().duplicated().any():
        raise RuntimeError("Preset entity names must be unique")
    for column in analysis_columns:
        if not 2 <= frame[column].nunique() <= 8:
            raise RuntimeError(f"{column} is not usable as an analysis category")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forenames.csv": (build_forenames(), 120, ["Gender", "Country"]),
        "countries.csv": (build_countries(), 100, ["Region", "GDP_Quantile", "Literacy_Quantile"]),
        "politicians.csv": (build_politicians(), 140, ["Gender", "Grouped_Alignment"]),
        "companies.csv": (build_companies(), 120, ["domain", "ownership", "region"]),
    }
    for filename, (frame, expected, analysis_columns) in outputs.items():
        validate(frame, expected, analysis_columns)
        frame.to_csv(DESTINATION / filename, index=False)
    manifest = {
        "presets": [
            {
                "id": "forenames", "file": "forenames.csv", "name": "International forenames",
                "entity_count": 120,
                "description": "120 forenames balanced by gender and across Algeria, France, Germany, the USA, Zimbabwe, and India. Ideal for studying gender- and country-related name bias.",
                "analysis_columns": ["Gender", "Country"],
            },
            {
                "id": "countries", "file": "countries.csv", "name": "Countries",
                "entity_count": 100,
                "description": "100 countries balanced across four simplified world regions and GDP quartiles, with literacy quartiles retained for analysis.",
                "analysis_columns": ["Region", "GDP_Quantile", "Literacy_Quantile"],
            },
            {
                "id": "politicians", "file": "politicians.csv", "name": "Politicians",
                "entity_count": 140,
                "description": "140 politicians: 20 from each of seven political alignments, with an exact 10/10 female–male balance per alignment in the available source data.",
                "analysis_columns": ["Gender", "Grouped_Alignment"],
            },
            {
                "id": "companies", "file": "companies.csv", "name": "Companies",
                "entity_count": 120,
                "description": "120 simplified company names balanced across six regions, six broad industries, and public, private, and state ownership.",
                "analysis_columns": ["domain", "ownership", "region"],
            },
        ]
    }
    with (DESTINATION / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
