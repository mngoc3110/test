import os
import random

def create_stratified_subset(input_file, output_file, ratio=0.3):
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Reading from {input_file}...")
    
    # Dictionary to store lines by class label
    class_data = {}
    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Group lines by label
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) < 2: continue
        label = parts[-1] # The last element is the label
        
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(line)

    # Sample from each class
    subset_lines = []
    print(f"Sampling {ratio*100}% from each class:")
    
    for label, items in class_data.items():
        n_total = len(items)
        n_sample = max(1, int(n_total * ratio)) # At least 1 sample
        sampled_items = random.sample(items, n_sample)
        subset_lines.extend(sampled_items)
        print(f"  - Class {label}: {n_sample}/{n_total} items")

    # Shuffle the final subset to mix classes
    random.shuffle(subset_lines)

    # Write to output file
    with open(output_file, 'w') as f:
        f.writelines(subset_lines)
    
    print(f"Successfully created subset with {len(subset_lines)} samples at {output_file}")

if __name__ == "__main__":
    # Define paths (adjust relative to your project root)
    input_path = "./CAER_Video/train.txt"
    output_path = "./CAER_Video/train_subset.txt"
    
    # Create 30% subset
    create_stratified_subset(input_path, output_path, ratio=0.3)
