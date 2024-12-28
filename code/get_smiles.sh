#!/bin/bash
# get name and smiles from ChEMBL for USAN/INN names
#
MAXLEN=256
MINLEN=5
psql -A --tuples-only -d chembl_34 <<EOF
select
  distinct
  lower(synonyms)as name , canonical_smiles
from 
  compound_structures,
  molecule_synonyms,
  molecule_dictionary  
where 
  compound_structures.molregno = molecule_synonyms.molregno 
  and compound_structures.molregno = molecule_dictionary.molregno 
  and ( syn_type = 'USAN' or syn_type = 'INN' )
  and canonical_smiles !~ 'Se|Te|As|Xe|Zn|Na|Ca|Cu|Mg|Li'
  and lower(synonyms) !~ ' .*ate$| .*ide$|[0-9]$*'  -- skip salt versions and nuclides
  and length(canonical_smiles) <= ${MAXLEN} 
  and length(canonical_smiles) >  ${MINLEN} 
EOF
