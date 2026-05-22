"""
config.py — Variable classification for the JGSS-Personas pipeline.

This file is the single source of truth for which JGSS-2017/2018 variables
play which role in the pipeline. The Random Forest pool is everything that
remains after these three lists are removed (see helper at the bottom).

Dataset: JGSS-2017/2018 Integrated Data (v1.0)
N respondents: 2,660 (744 from 2017 + 1,916 from 2018)
Total variables: 559

Methodology: Rupprecht et al. (2025), "German General Social Survey Personas",
arXiv:2511.21722. Adapted for JGSS with category structure tuned to the
items present in the 2017/2018 waves.

Author: [your name]
Last updated: 2026-05-19
"""


# ---------------------------------------------------------------------------
# 1. EXCLUDED VARIABLES
# ---------------------------------------------------------------------------
# Technical metadata, identifiers, interviewer observations, and the sampling
# weight. None of these enter the Random Forest as features, are included in
# personas, or are used as outcomes. The weight is loaded separately and
# applied only at the JSD evaluation step.
# ---------------------------------------------------------------------------

EXCLUDES = [
    # --- Identifiers ---
    "cumiduse",   # Cumulative ID for users
    "iduse",      # ID for users
    "ryear",      # Year of response (2017 vs 2018) — would leak wave identity

    # --- Date / time of interview ---
    "date", "month", "day",         # Interview date
    "dateb", "monthb", "dayb",      # Self-administered questionnaire collection date

    # --- Interviewer observations and process meta ---
    "coop",       # Respondent cooperation (interviewer rating)
    "ustand",     # Respondent understanding (interviewer rating)
    "area",       # Type of residential area (interviewer observation)
    "nameplat",   # Nameplate on home (interviewer observation)
    "tphouse",    # Type of residence detached (interviewer observation)
    "autolock",   # Auto-lock entrance (interviewer observation)
    "intercom",   # Type of intercom (interviewer observation)
    "proc",       # Survey process status
    "duration",   # Duration of interview
    "form",       # Form variant of self-administered questionnaire (split-ballot marker)

    # --- Sampling weight (used at evaluation only, never as a feature) ---
    "weight",

    # --- Redundant with age ---
    "dobyear",    # Date of birth: year — redundant with ageb, leaks cohort directly

    # --- Duplicates / redundant re-codings ---
    "marc",       # Marital Status — identical to domarry, which is already in CORE_DEMOGRAPHICS
    "ppjbxx15",   # Occupation of Father at Age 15 (local JP coding) — redundant with ppjxxe08 (ISCO-08)

    # --- Empty variables (zero valid responses in 2017/2018 wave) ---
    "cc04why",    # Child 04: Reason for Temporarily Leaving Home
    "cc05why",    # Child 05: Reason for Temporarily Leaving Home
    "cc06why",    # Child 06: Reason for Temporarily Leaving Home
    "cc07why",    # Child 07: Reason for Temporarily Leaving Home
    "cc08why",    # Child 08: Reason for Temporarily Leaving Home
    "cc09why",    # Child 09: Reason for Temporarily Leaving Home
    "ssxgrade",   # School Year (Spouse) — not collected in 2017/2018 wave
]


# ---------------------------------------------------------------------------
# 2. CORE DEMOGRAPHIC VARIABLES
# ---------------------------------------------------------------------------
# The fixed block included in every persona, regardless of TOP-k. These are
# the baseline socio-demographic anchors.
#
# Swap from German paper baseline: szincoma (respondent annual income) replaces
# domarry (marital status) as a fixed demographic. Marital status is moved to
# the RF pool — it emerged as a top-5 predictor (via its duplicate marc) so
# letting it compete data-driven is more principled. Income is fixed following
# the German GGSS Personas paper (Rupprecht et al. 2025).
#
# Note: BLOCK (regional block) and PREF (prefecture) are STRIPPED from the
# public JGSS-2017/2018 release. Only `size` (4-level municipality size) is
# available as a geographic proxy. Apply separately on JGSSDDS if regional
# data is needed.
# ---------------------------------------------------------------------------

CORE_DEMOGRAPHICS = [
    "sexa",       # Sex
    "ageb",       # Age (continuous)
    "size",       # Municipality size (4-level urban/rural proxy)
    "xxlstsch",   # Last school attended (education)
    "szincoma",   # Respondent annual income (overall, ordinal scale)
    "ccnumttl",   # Total number of children
    "xjob1wk",    # Work status (currently working / not) -- TO REVIEW!!
    "xxjob",      # Occupation
    "tpjbs",      # Spouse employment status
    "tp4wpla",    # Workplace category / type of establishment
]


# ---------------------------------------------------------------------------
# 3. OUTCOME VARIABLES (held out for LLM prediction + JSD evaluation)
# ---------------------------------------------------------------------------
# 19 variables across 6 thematic categories, mapped onto the German paper's
# 9 topic domains where the JGSS-2017/2018 questionnaire provides clean
# anchors. Every variable has a response rate >= 96%.
#
# Methodology note: these are HELD OUT — they do not enter the Random Forest
# importance ranking and are never included in any persona prompt.
# ---------------------------------------------------------------------------

OUTCOMES = {

    # Maps to German paper: Political Tendency + Social Capital
    "trust": [
        "op4trust",   # General trust in other people
        "tr3cgmnz",   # Trust in Diet members
        "tr3bcraz",   # Trust in ministries and government agencies
    ],

    # Maps to German paper: Ethnocentrism
    "ethnocentrism": [
        "qfnrincr",   # View on increasing the foreign population
        "q4samesm",   # View on same-sex marriage
        "op7gdevo",   # Estimation of human nature (good vs selfish)
    ],

    # Maps to German paper: Morality + Values & Life Goals
    "gender_family_norms": [
        "q7wwhhx",    # Gender role: men outside, women at home
        "q7jbmmcc",   # Mother working harms child (EASS scale)
        "q7mgcc",     # Marriage norm: married couple should have children
    ],

    # Maps to German paper: Social Inequality
    "social_inequality": [
        "q5gveqaa",   # Government should reduce income differences
        "opincdif",   # Income inequality has become too large
        "opnucpol",   # Opinion on nuclear power policy
    ],

    # Maps to German paper: Lifestyle (subjective wellbeing proxy)
    # NOTE: 4 variables here (others have 3) — op5happz added by user request
    # to compare "feel" measure (happiness) vs "judge" measure (life satisfaction).
    "wellbeing": [
        "stalllf",    # Overall life satisfaction (5-point)
        "nofutr",     # Hopelessness: no hope for a better future
        "sfmhdprs",   # Felt downhearted / depressed in past 4 weeks (SF-12)
        "op5happz",   # Degree of happiness (5-point)
    ],

    # Maps to German paper: Social Capital
    "community_civic": [
        "opnbmtcn",   # Neighbors are mutually concerned for each other
        "wllive",     # Wish to keep living in the same area
        "mempltgp",   # Membership in political associations
    ],
}


# ---------------------------------------------------------------------------
# Convenience helpers (use these in pipeline.py to avoid manual flattening)
# ---------------------------------------------------------------------------

# Flat list of all 19 outcome variables, no category nesting
OUTCOMES_FLAT = [var for cat in OUTCOMES.values() for var in cat]

# Codes treated as missing across all variables (JGSS conventions:
# 9 = "No answer" for 1-digit items, 99 for 2-digit, 999 for 3-digit).
# Note: "Don't know" is coded distinctly per question and is NOT auto-removed
# here — for some questions DK is substantively meaningful and you may want
# to keep it as a valid category.
MISSING_CODES = [9, 99, 999]


def get_rf_pool(all_variables):
    """
    Return the Random Forest pool: every variable not in EXCLUDES,
    CORE_DEMOGRAPHICS, or OUTCOMES_FLAT.

    Parameters
    ----------
    all_variables : list[str] or pandas.Index
        The full list of variable names in the loaded SAV file.

    Returns
    -------
    list[str]
        Variables eligible for the TOP-k importance ranking.
    """
    classified = set(EXCLUDES) | set(CORE_DEMOGRAPHICS) | set(OUTCOMES_FLAT)
    return [v for v in all_variables if v not in classified]


def sanity_check(all_variables):
    """
    Run after loading the SAV. Confirms that:
      - every variable named in EXCLUDES / CORE / OUTCOMES exists in the data
      - no variable is double-classified
      - prints the resulting bucket sizes

    Call this once at the start of the pipeline; fail loudly on mismatches.
    """
    av = set(all_variables)

    # Existence checks
    for bucket_name, bucket in [("EXCLUDES", EXCLUDES),
                                 ("CORE_DEMOGRAPHICS", CORE_DEMOGRAPHICS),
                                 ("OUTCOMES_FLAT", OUTCOMES_FLAT)]:
        missing = [v for v in bucket if v not in av]
        if missing:
            raise ValueError(
                f"{bucket_name} contains variables not in the dataset: {missing}"
            )

    # Overlap checks
    overlaps = [
        ("EXCLUDES & CORE_DEMOGRAPHICS", set(EXCLUDES) & set(CORE_DEMOGRAPHICS)),
        ("EXCLUDES & OUTCOMES",          set(EXCLUDES) & set(OUTCOMES_FLAT)),
        ("CORE_DEMOGRAPHICS & OUTCOMES", set(CORE_DEMOGRAPHICS) & set(OUTCOMES_FLAT)),
    ]
    for name, overlap in overlaps:
        if overlap:
            raise ValueError(f"Double-classification in {name}: {overlap}")

    rf = get_rf_pool(all_variables)
    print("Variable classification check passed:")
    print(f"  Total in dataset:    {len(all_variables)}")
    print(f"  Excluded:            {len(EXCLUDES)}")
    print(f"  Core demographics:   {len(CORE_DEMOGRAPHICS)}")
    print(f"  Outcomes:            {len(OUTCOMES_FLAT)} across {len(OUTCOMES)} categories")
    print(f"  RF pool (the rest):  {len(rf)}")

    total = len(EXCLUDES) + len(CORE_DEMOGRAPHICS) + len(OUTCOMES_FLAT) + len(rf)
    assert total == len(all_variables), (
        f"Bucket sizes sum to {total}, expected {len(all_variables)}"
    )