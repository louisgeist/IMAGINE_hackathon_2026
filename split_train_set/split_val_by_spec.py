#!/usr/bin/env python3
"""
Separate images specified in minival_data_def.txt from train partition into minival partition.

This script can operate in two modes:
1. SPLIT mode: Move specified images from train to minival partition
2. VERIFY mode: Check if the minival partition exactly matches the specification in minival_data_def.txt
"""

import shutil
from pathlib import Path
from collections import defaultdict

# ==== CONFIG ====
TRAIN_DIR = Path("../data/train")
MINIVAL_DIR = Path("../data/val")
SPEC_FILE = Path("./val_data_def.txt")
TRAIN_COUNTS_FILE = Path("./train_data_counts.txt")

# Set to True to verify minival matches spec, False to perform the split
VERIFY_MODE = True

# =================

def parse_spec_file(spec_path):
    """
    Read the spec file and return a set of filenames to move.
    Also returns a dict mapping class_id -> set of filenames in that class.
    """
    filenames = set()
    class_files = defaultdict(set)
    
    with open(spec_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filenames.add(line)
            
            # Extract class ID (prefix before first underscore followed by digits)
            # e.g., "n01440764_11566.JPEG" -> class_id is "n01440764"
            parts = line.rsplit('_', 1)  # split from right to handle numbers
            if len(parts) == 2:
                class_id = parts[0]
                class_files[class_id].add(line)
    
    return filenames, class_files


def parse_train_counts_file(counts_path):
    """
    Read the train counts file and return a dict mapping class_id -> expected count.
    Format: class_id: count (one per line)
    """
    class_counts = {}
    
    with open(counts_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            class_id, count_str = line.split(':', 1)
            class_id = class_id.strip()
            try:
                count = int(count_str.strip())
                class_counts[class_id] = count
            except ValueError:
                print(f"WARNING: Could not parse count for {class_id}: {count_str}")
    
    return class_counts


def split_minival(train_dir, minival_dir, spec_path):
    """
    Move images specified in spec_path from train to minival partition.
    """
    filenames, class_files = parse_spec_file(spec_path)
    
    print(f"Read {len(filenames)} images from {spec_path}")
    print(f"Spanning {len(class_files)} classes")
    print()
    
    # Create minival directory if it doesn't exist
    minival_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    missing_count = 0
    
    for class_id, files_in_class in sorted(class_files.items()):
        train_class_dir = train_dir / class_id
        minival_class_dir = minival_dir / class_id
        
        if not train_class_dir.exists():
            print(f"WARNING: Class directory {class_id} not found in train")
            missing_count += len(files_in_class)
            continue
        
        # Create class directory in minival
        minival_class_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in files_in_class:
            src_path = train_class_dir / filename
            dst_path = minival_class_dir / filename
            
            if not src_path.exists():
                print(f"WARNING: File not found {src_path}")
                missing_count += 1
                continue
            
            # Move the file
            shutil.move(str(src_path), str(dst_path))
            moved_count += 1
        
        print(f"{class_id}: {len(files_in_class)} images moved")
    
    print()
    print(f"Total moved: {moved_count}")
    if missing_count > 0:
        print(f"Total missing: {missing_count}")
    print("Done!")


def verify_minival(minival_dir, spec_path, train_dir=None, train_counts_path=None):
    """
    Verify that the minival partition exactly matches the specification.
    Optionally also checks that minival files are NOT in the train partition.
    Optionally checks that train partition file counts match expected counts.
    """
    filenames_spec, class_files_spec = parse_spec_file(spec_path)
    
    print(f"Spec requires {len(filenames_spec)} images in {len(class_files_spec)} classes")
    print()
    
    # Collect all files currently in minival
    minival_files = set()
    if minival_dir.exists():
        for class_dir in minival_dir.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.is_file():
                        minival_files.add(img_file.name)
    
    print(f"Found {len(minival_files)} images in minival partition")
    print()
    
    # Check for missing files
    missing = filenames_spec - minival_files
    extra = minival_files - filenames_spec
    
    if missing:
        print(f"MISSING {len(missing)} files:")
        for f in sorted(missing)[:20]:
            print(f"  - {f}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        print()
    
    if extra:
        print(f"EXTRA {len(extra)} files:")
        for f in sorted(extra)[:20]:
            print(f"  - {f}")
        if len(extra) > 20:
            print(f"  ... and {len(extra) - 20} more")
        print()
    
    # Check that minival files are NOT in train partition
    duplicates = set()
    if train_dir and train_dir.exists():
        print("Checking for duplicates in train partition...")
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.is_file() and img_file.name in minival_files:
                        duplicates.add(img_file.name)
    
    if duplicates:
        print(f"✗ DUPLICATE {len(duplicates)} files found in both minival and train:")
        for f in sorted(duplicates)[:20]:
            print(f"  - {f}")
        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more")
        print()
    
    # Check train partition file counts
    train_counts_mismatch = False
    if train_counts_path and train_counts_path.exists():
        print("Checking train partition file counts...")
        expected_counts = parse_train_counts_file(train_counts_path)
        
        # Count files in train partition per class
        actual_counts = defaultdict(int)
        if train_dir and train_dir.exists():
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir():
                    class_id = class_dir.name
                    file_count = sum(1 for _ in class_dir.iterdir() if _.is_file())
                    actual_counts[class_id] = file_count
        
        print(f"Expected {len(expected_counts)} classes in train partition")
        print(f"Found {len(actual_counts)} classes in train partition")
        print()
        
        # Compare counts
        mismatches = []
        for class_id in sorted(expected_counts.keys()):
            expected = expected_counts[class_id]
            actual = actual_counts.get(class_id, 0)
            if expected != actual:
                mismatches.append((class_id, expected, actual))
        
        if mismatches:
            train_counts_mismatch = True
            print(f"✗ MISMATCH: {len(mismatches)} classes have incorrect file counts:")
            for class_id, expected, actual in mismatches[:20]:
                print(f"  - {class_id}: expected {expected}, found {actual}")
            if len(mismatches) > 20:
                print(f"  ... and {len(mismatches) - 20} more")
            print()
        else:
            print("✓ All train partition file counts match expected values")
            print()
    
    if not missing and not extra and not duplicates and not train_counts_mismatch:
        print("✓ VERIFIED: minival partition exactly matches specification!")
        print("✓ No duplicate files found in train partition")
        print("✓ Train partition file counts are correct")
        return True
    else:
        print("✗ MISMATCH: minival partition or train partition does not match specification")
        return False


if __name__ == "__main__":
    if not SPEC_FILE.exists():
        print(f"ERROR: Spec file not found: {SPEC_FILE}")
        exit(1)
    
    if VERIFY_MODE:
        print("=== VERIFY MODE ===")
        print()
        verify_minival(MINIVAL_DIR, SPEC_FILE, TRAIN_DIR, TRAIN_COUNTS_FILE)
    else:
        print("=== SPLIT MODE ===")
        print()
        split_minival(TRAIN_DIR, MINIVAL_DIR, SPEC_FILE)
