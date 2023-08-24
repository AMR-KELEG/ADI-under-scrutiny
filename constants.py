NADI_FILE_PATH = "data/NADI_datasets/NADI2021_2023.tsv"
MADAR_FILE_PATH = "data/MADAR.tsv"

# Dialects covered in NADI2023
DIALECTS = [
    "Algeria",
    "Bahrain",
    "Egypt",
    "Iraq",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Sudan",
    "Syria",
    "Tunisia",
    "UAE",
    "Yemen",
]

DIALECTS_INDEX_MAP = {d: i for i, d in enumerate(DIALECTS)}
DIALECTS_INDEX_INVERTED_MAP = {i: d for i, d in enumerate(DIALECTS)}

COUNTRY_TO_REGION = {
    "Algeria": "maghreb",
    "Bahrain": "gulf",
    "Egypt": "nile_basin",
    "Iraq": "gulf",
    "Jordan": "levant",
    "Kuwait": "gulf",
    "Lebanon": "levant",
    "Libya": "maghreb",
    "Morocco": "maghreb",
    "Oman": "gulf",
    "Palestine": "levant",
    "Qatar": "gulf",
    "Saudi_Arabia": "gulf",
    "Sudan": "nile_basin",
    "Syria": "levant",
    "Tunisia": "maghreb",
    "UAE": "gulf",
    "Yemen": "gulf_aden",
}

MADAR_CITY_TO_COUNTRY = {
    "ALE": "Syria",
    "ALG": "Algeria",
    "ALX": "Egypt",
    "AMM": "Jordan",
    "ASW": "Egypt",
    "BAG": "Iraq",
    "BAS": "Iraq",
    "BEI": "Lebanon",
    "BEN": "Libya",
    "CAI": "Egypt",
    "DAM": "Syria",
    "DOH": "Qatar",
    "FES": "Morocco",
    "JED": "Saudi_Arabia",
    "JER": "Palestine",
    "KHA": "Sudan",
    "MOS": "Iraq",
    "MUS": "Oman",
    "RAB": "Morocco",
    "RIY": "Saudi_Arabia",
    "SAL": "Jordan",
    "SAN": "Yemen",
    "SFX": "Tunisia",
    "TRI": "Libya",
    "TUN": "Tunisia",
}

MultiDialect_COUNTRYCODE_TO_COUNTRY = {
    "EG": "Egypt",
    "TN": "Tunisia",
    "SY": "Syria",
    "JO": "Jordan",
    "PA": "Palestine",
}

COUNTRIES_IN_SAME_REGION = {
    dialect: [
        other_dialect
        for other_dialect in DIALECTS
        if other_dialect != dialect
        and COUNTRY_TO_REGION[dialect] == COUNTRY_TO_REGION[other_dialect]
    ]
    for dialect in DIALECTS
}

COUNTRIES_IN_SAME_REGION_int = {
    DIALECTS_INDEX_MAP[dialect]: [
        DIALECTS_INDEX_MAP[other_dialect]
        for other_dialect in COUNTRIES_IN_SAME_REGION[dialect]
    ]
    for dialect in DIALECTS
}
