NADI_FILE_PATH="data/NADI2021_2023.tsv"

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
