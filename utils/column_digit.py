import pandas as pd
import numpy as np
import os
import math
import shutil
from tabulate import tabulate  # Install with: pip install tabulate

def count_digits(value):
    """Count integer and decimal digits in a numeric value"""
    if pd.isna(value):
        return 0, 0  # No digits for NaN values
    
    try:
        # Convert to string and split by decimal point
        str_val = str(float(value))
        if 'e' in str_val.lower():  # Handle scientific notation
            base, exp = str_val.lower().split('e')
            base_int, base_dec = base.split('.')
            exp_val = int(exp)
            if exp_val > 0:
                int_digits = len(base_int) + exp_val
                dec_digits = max(0, len(base_dec) - exp_val)
            else:
                int_digits = max(0, len(base_int) + exp_val)
                dec_digits = len(base_dec) - exp_val
            return int_digits, dec_digits
            
        parts = str_val.split('.')
        
        # Count digits before decimal (excluding leading zeros and minus sign)
        int_part = parts[0]
        if int_part.startswith('-'):
            int_part = int_part[1:]  # Remove minus sign
        int_digits = len(int_part.lstrip('0')) or (1 if int_part == '0' else 0)
        
        # Count digits after decimal (excluding trailing zeros)
        dec_digits = 0
        if len(parts) > 1:
            dec_digits = len(parts[1].rstrip('0'))
            
        return int_digits, dec_digits
    except:
        return 0, 0  # Return 0 for non-numeric values

def analyze_column_digits(df, column):
    """Analyze digit distribution in a column"""
    if pd.api.types.is_numeric_dtype(df[column]):
        # Only analyze numeric columns
        int_digits = []
        dec_digits = []
        
        for value in df[column]:
            i_dig, d_dig = count_digits(value)
            int_digits.append(i_dig)
            dec_digits.append(d_dig)
        
        return {
            'column': column,
            'avg_int_digits': np.mean(int_digits),
            'max_int_digits': max(int_digits),
            'min_int_digits': min(int_digits),
            'avg_dec_digits': np.mean(dec_digits),
            'max_dec_digits': max(dec_digits),
            'min_dec_digits': min(dec_digits),
            'numeric': True,
            'dtype': str(df[column].dtype),
            'unique_values': df[column].nunique()
        }
    else:
        # For non-numeric columns, count characters
        lengths = [len(str(x)) for x in df[column]]
        return {
            'column': column,
            'avg_length': np.mean(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'numeric': False,
            'dtype': str(df[column].dtype),
            'unique_values': df[column].nunique()
        }

def analyze_dataset_digits(filepath):
    """Analyze digit distribution for all columns in a dataset"""
    # Load the dataset
    print(f"Loading dataset: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    
    # Analyze each column
    results = []
    for column in df.columns:
        results.append(analyze_column_digits(df, column))
    
    # Print numeric column results
    numeric_results = [r for r in results if r['numeric']]
    if numeric_results:
        print("\n=== Numeric Column Digit Analysis ===")
        table_data = []
        for r in numeric_results:
            table_data.append([
                r['column'], 
                f"{r['avg_int_digits']:.1f}",
                r['min_int_digits'],
                r['max_int_digits'],
                f"{r['avg_dec_digits']:.1f}",
                r['min_dec_digits'],
                r['max_dec_digits'],
                r['unique_values']
            ])
        
        headers = ["Column", "Avg Int", "Min Int", "Max Int", "Avg Dec", "Min Dec", "Max Dec", "Unique"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print non-numeric column results
    non_numeric_results = [r for r in results if not r['numeric']]
    if non_numeric_results:
        print("\n=== Non-Numeric Column Analysis ===")
        table_data = []
        for r in non_numeric_results:
            table_data.append([
                r['column'], 
                f"{r['avg_length']:.1f}",
                r['min_length'],
                r['max_length'],
                r['dtype'],
                r['unique_values']
            ])
        
        headers = ["Column", "Avg Length", "Min Length", "Max Length", "Data Type", "Unique"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Additional summary statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    return results

def group_and_save_columns_by_digits(df, results, output_dir="digit"):
    """Group columns by average integer digits and save to separate files"""
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove if exists to start fresh
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Grouping columns by integer digits and saving to '{output_dir}/' ===")
    
    # Group columns by average integer digits (rounded)
    digit_groups = {}
    for r in results:
        if r['numeric']:
            # Round to integer
            int_digits = round(r['avg_int_digits'])
            if int_digits not in digit_groups:
                digit_groups[int_digits] = []
            digit_groups[int_digits].append(r['column'])
    
    # Create a summary file
    with open(os.path.join(output_dir, "00_summary.txt"), "w") as f:
        f.write("Column Grouping by Integer Digits\n")
        f.write("===============================\n\n")
        
        for digits, columns in sorted(digit_groups.items()):
            f.write(f"Group {digits} digits: {', '.join(columns)}\n")
            f.write(f"File: group_{digits}_digits.csv\n\n")
    
    # Save each group to a separate file
    for digits, columns in digit_groups.items():
        if columns:  # Only save non-empty groups
            # Select just these columns from the dataframe
            subset_df = df[columns]
            # Save to CSV
            output_file = os.path.join(output_dir, f"group_{digits}_digits.csv")
            subset_df.to_csv(output_file, index=False)
            print(f"Saved {len(columns)} columns with ~{digits} integer digits to {output_file}")
    
    # Also save a file with timestamp column (if exists) plus each group
    timestamp_col = None
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col:
        print(f"\nFound possible timestamp column: {timestamp_col}")
        for digits, columns in digit_groups.items():
            if columns:  # Only save non-empty groups
                # Add timestamp column at the beginning
                ts_columns = [timestamp_col] + [c for c in columns if c != timestamp_col]
                subset_df = df[ts_columns]
                output_file = os.path.join(output_dir, f"group_{digits}_digits_with_timestamp.csv")
                subset_df.to_csv(output_file, index=False)
                print(f"Saved {timestamp_col} + {len(columns)} columns to {output_file}")

if __name__ == "__main__":
    # Ask for dataset path
    default_path = "./datasets/network/processed/snmp_first10.csv"
    
    filepath = input(f"Enter dataset path (default: {default_path}): ")
    if not filepath.strip():
        filepath = default_path
    
    if not os.path.exists(filepath):
        # Try with project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        alternative_path = os.path.join(project_root, filepath)
        if os.path.exists(alternative_path):
            filepath = alternative_path
        else:
            print(f"Error: File not found at {filepath} or {alternative_path}")
            exit(1)
            
    # Analyze the dataset
    results = analyze_dataset_digits(filepath)
    
    # Ask if user wants to group columns
    group_columns = input("\nDo you want to group columns by integer digits and save to separate files? (y/n): ").lower().startswith('y')
    
    if group_columns:
        # Get output directory
        output_dir = input("Enter output directory name (default: digit): ").strip() or "digit"
        
        # Create full path for output directory
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(filepath), output_dir)
        
        # Load the dataset again to access the dataframe
        df = pd.read_csv(filepath)
        
        # Group and save columns
        group_and_save_columns_by_digits(df, results, output_dir)
        
        print(f"\nDone! Files saved to {output_dir}/")