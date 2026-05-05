# extract_pockets_around_ligands.py
import os
import numpy as np

def extract_pocket_around_ligands(input_pdb, output_pdb, max_atoms=450, distance_cutoff=8.0):
    """
    Extract protein atoms around non-water HETATM (ligands).
    """
    print(f"Processing {input_pdb}...")
    
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    # Find ligand atoms (non-water HETATM)
    ligand_atoms = []
    ligand_coords = []
    
    for line in lines:
        if line.startswith("HETATM"):
            res_name = line[17:20].strip()
            # Skip water and common ions/salts
            if res_name not in ['HOH', 'WAT', 'SO4', 'PO4', 'CL', 'NA', 'MG', 'CA', 'K', 'ZN']:
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ligand_atoms.append(line)
                    ligand_coords.append([x, y, z])
                except ValueError:
                    continue
    
    print(f"  Found {len(ligand_atoms)} ligand atoms")
    
    # If no ligands found, use alternative method
    if not ligand_coords:
        print(f"  No ligands found, using geometric center")
        # Use center of protein
        protein_coords = []
        for line in lines:
            if line.startswith("ATOM"):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    protein_coords.append([x, y, z])
                except ValueError:
                    continue
        
        if protein_coords:
            center = np.mean(protein_coords, axis=0)
            ligand_coords = [center]
        else:
            print(f"  ERROR: No coordinates found in {input_pdb}")
            return
    
    # Find protein atoms near ligands
    pocket_atoms = []
    pocket_coords = []
    
    for line in lines:
        if line.startswith("ATOM"):
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                # Calculate minimum distance to any ligand atom
                min_dist = float('inf')
                for lx, ly, lz in ligand_coords:
                    dist = np.sqrt((x-lx)**2 + (y-ly)**2 + (z-lz)**2)
                    min_dist = min(min_dist, dist)
                
                if min_dist <= distance_cutoff:
                    pocket_atoms.append(line)
                    pocket_coords.append([x, y, z, min_dist])  # Store with distance
            except ValueError:
                continue
    
    print(f"  Found {len(pocket_atoms)} protein atoms within {distance_cutoff}Å of ligands")
    
    # If too many atoms, take closest ones
    if len(pocket_atoms) > max_atoms:
        print(f"  Too many atoms ({len(pocket_atoms)}), selecting {max_atoms} closest")
        # Sort by distance to ligands
        pocket_atoms_sorted = sorted(zip(pocket_atoms, [c[3] for c in pocket_coords]), key=lambda x: x[1])
        pocket_atoms = [atom for atom, dist in pocket_atoms_sorted[:max_atoms]]
    
    # Write output file
    with open(output_pdb, 'w') as f:
        # Add CRYST1 line if present in original
        for line in lines:
            if line.startswith("CRYST1"):
                f.write(line)
                break
        
        # Write pocket atoms
        for atom in pocket_atoms:
            f.write(atom)
        
        # Write END
        f.write("END\n")
    
    print(f"  Created {output_pdb} with {len(pocket_atoms)} atoms")
    return len(pocket_atoms)

# Process all files
input_dir = "datasets/new_data"
output_dir = "datasets/new_data_processed"

os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Extracting pockets around ligands...")
print("=" * 60)

atom_counts = {}
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".pdb"):
        input_path = os.path.join(input_dir, filename)
        output_name = filename.replace(".pdb", "_pocket.pdb")
        output_path = os.path.join(output_dir, output_name)
        
        count = extract_pocket_around_ligands(
            input_path, 
            output_path, 
            max_atoms=450,  # Keep under 500 limit
            distance_cutoff=8.0  # 8Å radius around ligands
        )
        atom_counts[filename] = count
        print()

print("=" * 60)
print("Summary:")
print("=" * 60)
for filename, count in atom_counts.items():
    print(f"{filename:30} -> {count:4d} atoms")
