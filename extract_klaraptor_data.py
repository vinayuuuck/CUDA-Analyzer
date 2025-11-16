import re
import pandas as pd
from pathlib import Path

def parse_kernel_table_file(filepath):
    """
    Parse kernel_*_table_*.txt files
    Format:
    [N: 128]
    [b0: 2]
    [b1: 16]
    [Exec_cycles_app: 22608601.2254]
    ...
    """
    data = []
    current_entry = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Match pattern: [key: value]
            match = re.match(r'\[([^:]+):\s*([^\]]+)\]', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Store relevant fields
                if key == 'N':
                    # New entry starting
                    if current_entry and 'N' in current_entry:
                        data.append(current_entry.copy())
                    current_entry = {'N': int(value)}
                    
                elif key == 'b0':
                    current_entry['bx'] = int(value)
                elif key == 'b1':
                    current_entry['by'] = int(value)
                elif key == 'Exec_cycles_app':
                    current_entry['exec_cycles'] = float(value)
                elif key == 'Threads_per_block':
                    current_entry['threads_per_block'] = int(value)
                elif key == 'occupancy':
                    current_entry['occupancy'] = float(value)
    
    # Add last entry
    if current_entry and 'N' in current_entry:
        data.append(current_entry)
    
    return data

def parse_kernel_results_file(filepath):
    """
    Parse kernel_*_results_*.txt files
    Format: [N:32, b0:1, b1:32, mwp: 3, cwp: 3, clocks:707234.2857, ...]
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Match the bracket format
            match = re.search(r'\[N:(\d+),\s*b0:(\d+)\s*,\s*b1:(\d+)\s*,.*?clocks:([\d.]+)', line)
            if match:
                data.append({
                    'N': int(match.group(1)),
                    'bx': int(match.group(2)),
                    'by': int(match.group(3)),
                    'exec_cycles': float(match.group(4))
                })
    
    return data

def extract_all_klaraptor_data(base_dir='./KLARAPTORresults'):
    """Extract all timing data from KLARAPTOR results"""
    
    base_path = Path(base_dir)
    all_data = []
    
    kernel_dirs = sorted([d for d in base_path.iterdir() 
                         if d.is_dir() and d.name.startswith('polybench_')])
    
    print(f"Found {len(kernel_dirs)} kernel directories\n")
    
    for kernel_dir in kernel_dirs:
        kernel_name = kernel_dir.name.replace('polybench_', '')
        
        print(f"Processing {kernel_name}...", end=' ')
        
        # Try kernel_table files first (more detailed)
        table_files = list(kernel_dir.glob('kernel_*_table_*.txt'))
        
        timing_data = []
        
        if table_files:
            # Use the first table file found
            try:
                timing_data = parse_kernel_table_file(table_files[0])
                if timing_data:
                    print(f"✓ {len(timing_data)} samples (from table)")
            except Exception as e:
                print(f"Table parse failed: {e}", end=' ')
        
        # If no table data, try results files
        if not timing_data:
            results_files = list(kernel_dir.glob('kernel_*_results_*.txt'))
            if results_files:
                try:
                    timing_data = parse_kernel_results_file(results_files[0])
                    if timing_data:
                        print(f"✓ {len(timing_data)} samples (from results)")
                except Exception as e:
                    print(f"Results parse failed: {e}")
        
        # Add kernel name and default values
        for row in timing_data:
            row['kernel'] = kernel_name
            row['bz'] = 1
            if 'bx' in row and 'by' in row:
                row['total_threads'] = row['bx'] * row['by']
            
            # Convert cycles to approximate seconds
            # Assume ~1.5 GHz GPU clock (adjust if needed)
            if 'exec_cycles' in row:
                gpu_freq_ghz = 1.5  # Approximate
                row['exec_time'] = row['exec_cycles'] / (gpu_freq_ghz * 1e9)
            
            all_data.append(row)
        
        if not timing_data:
            print("✗ No data found")
    
    if not all_data:
        print("\n⚠️  No data extracted!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Keep only essential columns
    essential_cols = ['kernel', 'N', 'bx', 'by', 'bz', 'total_threads', 'exec_time']
    optional_cols = ['occupancy', 'threads_per_block']
    
    cols_to_keep = [c for c in essential_cols if c in df.columns]
    cols_to_keep += [c for c in optional_cols if c in df.columns]
    
    df = df[cols_to_keep]
    
    # Save in current directory
    output_file = 'klaraptor_timing_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ SUCCESS! Extracted KLARAPTOR data")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"Kernels: {df['kernel'].nunique()}")
    print(f"Saved to: {output_file}\n")
    
    print("Breakdown by kernel:")
    print(df.groupby('kernel').size())
    
    if 'N' in df.columns:
        print(f"\nProblem sizes: {sorted(df['N'].unique())}")
    if 'exec_time' in df.columns:
        print(f"Execution time range: {df['exec_time'].min():.6f}s - {df['exec_time'].max():.6f}s")
    
    return df

if __name__ == '__main__':
    df = extract_all_klaraptor_data()
    
    if df is not None:
        print(f"\nFirst 10 samples:")
        print(df.head(10))
        
        print(f"\nDataset info:")
        print(df.info())
