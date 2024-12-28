#!/bin/bash
# query to create list of smiles from postgresql ChEMBL

psql -A  --tuples-only -d chembl_33 -c "
select
  distinct
  canonical_smiles 
from 
  molecule_dictionary, compound_structures  
where 
  compound_structures.molregno = molecule_dictionary.molregno 
  and usan_stem != '' 
  and molecule_type = 'Small molecule' 
"
