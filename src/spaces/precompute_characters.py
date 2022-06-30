import json

from so import SO, SOCharacterDenominatorFree
from su import SU, SUCharacterDenominatorFree

# set to True to recalculate all characters, set to False to add to the already existing without recalculating
recalculate = False
storage_file_name = 'precomputed_characters.json'

groups = [
    (SO, 3, SOCharacterDenominatorFree),
    (SO, 4, SOCharacterDenominatorFree),
    (SO, 5, SOCharacterDenominatorFree),
    (SO, 6, SOCharacterDenominatorFree),
    (SU, 3, SUCharacterDenominatorFree),
    (SU, 4, SUCharacterDenominatorFree),
    (SU, 5, SUCharacterDenominatorFree),
]
# the number of representations to be calculated for each group
order = 10

characters = {}
if not recalculate:
    with open(storage_file_name, 'r') as file:
        characters = json.load(file)

for group_type, dim, character_class in groups:
    group = group_type(n=dim, order=order)
    group_name = '{}({})'.format(group_type.__name__, dim)
    print(group_name)
    if recalculate or (not recalculate and group_name not in characters):
        characters[group_name] = {}
    for irrep in group.lb_eigenspaces:
        if str(irrep.index) not in characters[group_name]:
            character = character_class(representation=irrep, precomputed=False)
            coeffs, monoms = character._compute_character_formula()
            print(irrep.index, coeffs, monoms)
            characters[group_name][str(irrep.index)] = (coeffs, monoms)

with open(storage_file_name, 'w') as file:
    json.dump(characters, file, sort_keys=True)
