# normalize smiles strings
# remove isotopic data
# must only contain organic elements
#
import sys
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

MAX_LEN=256
MIN_CARBON = 2
unique = set()

for rsmiles in sys.stdin:

    rsmiles = rsmiles.strip()

    for smiles in rsmiles.split('.'):

        h = hash(smiles)
        if h in unique:
            continue

        try:
            mol = Chem.MolFromSmiles(smiles, sanitize = True)
            reject = False
            if mol is not None:
                for atom in mol.GetAtoms():
                    symbol  = atom.GetSymbol()
                    if symbol not in ['C', 'P', 'N', 'O', 'S', 'H', 'F', 'Cl', 'Br', 'I' ]:
                         reject = True
                         break
                    atom.SetIsotope(0)
   
                if reject:
                   continue
 
                fixed = Chem.MolToSmiles(mol, isomericSmiles = True, canonical = True)
                fh = hash(fixed) 
                if fixed != '' and fixed.lower().count('c') > MIN_CARBON  and fh not in unique 
                  and len(fixed) <= MAX_LEN:
                    print(fixed)
                    unique.add(fh)    
                
        except:
            unique.add(h)

