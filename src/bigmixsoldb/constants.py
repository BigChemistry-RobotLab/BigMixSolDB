from __future__ import annotations

import re

OUTPUT_COLUMNS = [
    "Compound Name",
    "Synonyms",
    "SMILES_Compound",
    "Solvent 1",
    "SMILES_Solvent_1",
    "Solvent 2",
    "SMILES_Solvent_2",
    "Solvent 3",
    "SMILES_Solvent_3",
    "Extra Solvents",
    "Concentration Solvent 1",
    "Concentration Solvent 2",
    "Concentration Solvent 3",
    "Concentration Unit",
    "Solubility",
    "Solubility Unit",
    "Temperature",
    "Temperature Unit",
    "Pressure",
    "Pressure Unit",
    "Requires Review",
    "doi",
]

FRACTION_UNITS = {"mole fraction", "mass fraction", "volume fraction"}

GENERIC_NAME_PATTERNS = [
    re.compile(r"^form\s+[ivx0-9]+$", re.IGNORECASE),
    re.compile(r"^polymorph\s+[ivx0-9]+$", re.IGNORECASE),
    re.compile(r"^compound\s+[a-z0-9]+$", re.IGNORECASE),
    re.compile(r"^sample\s+[a-z0-9]+$", re.IGNORECASE),
    re.compile(r"^phase\s+[ivx0-9]+$", re.IGNORECASE),
    re.compile(r"^crystal\s+[ivx0-9]+$", re.IGNORECASE),
    re.compile(r"^modification\s+[ivx0-9]+$", re.IGNORECASE),
    re.compile(r"^structure\s+[ivx0-9]+$", re.IGNORECASE),
    re.compile(r"^isomer\s+[a-z0-9]+$", re.IGNORECASE),
]

CAS_PATTERN = re.compile(r"(?<!\d)(\d{2,7}-\d{2}-\d)(?!\d)")

MISSING_VALUES = {"", "-", "none", "nan", "n/a"}
