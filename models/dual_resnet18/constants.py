"""Shared label conventions for the dual-head model. Every other file imports these; never reorder them.

Follows the team rule in models/README.md (F1 is scored on class 1, the rarer class):
    age:    0 adult, 1 young
    antler: 0 antlerless, 1 antlered
"""

ADULT, YOUNG = 0, 1
ANTLERLESS, ANTLERED = 0, 1

AGE_NAMES = ('adult', 'young')
ANTLER_NAMES = ('antlerless', 'antlered')

AGE_MAP = {'ADULT': ADULT, 'YOUNG': YOUNG}
ANTLER_MAP = {'ANTLERLESS': ANTLERLESS, 'ANTLERED': ANTLERED}

# Joint table is indexed [antler, age]; antlered + young is biologically impossible
IMPOSSIBLE_CELL = (ANTLERED, YOUNG)

# Combined label for an (antler, age) cell of the joint table, plus the age-only fallback
COMBINED_NAMES = {
    (ANTLERLESS, ADULT): 'adult_antlerless',
    (ANTLERED, ADULT): 'adult_antlered',
    (ANTLERLESS, YOUNG): 'young_antlerless',
}
ADULT_UNDETERMINED = 'adult_undetermined'
