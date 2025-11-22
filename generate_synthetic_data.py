#!/usr/bin/env python3
"""
Generate synthetic vector data in TSV and binary formats for SPTAG testing.
Creates a folder with files containing zero vectors in the specified format.
"""
import argparse
import struct
import os
import numpy as np
from typing import Optional


def create_tsv_file(output_dir: str, num_vectors: int, dimensions: int, data_type: str) -> str:
    """
    Create a TSV file with synthetic zero vectors.
    
    Args:
        output_dir: Output directory path
        num_vectors: Number of vectors to generate
        dimensions: Vector dimensionality
        data_type: Data type ('float' or 'int8')
    
    Returns:
        Path to created TSV file
    """
    tsv_file = os.path.join(output_dir, f"synthetic_{num_vectors}x{dimensions}_{data_type}.tsv")
    
    print(f"Creating TSV file: {tsv_file}")
    print(f"Generating {num_vectors} vectors with {dimensions} dimensions of type {data_type}")
    
    with open(tsv_file, 'w') as f:
        for i in range(num_vectors):
            # Generate a synthetic ID (32-character hex string like in the original)
            vector_id = f"{i:032X}"
            
            # Create zero vector
            if data_type == 'float':
                vector_values = ['0.0'] * dimensions
            else:  # int8
                vector_values = ['0'] * dimensions
            
            # Join with pipe separator
            vector_str = '|'.join(vector_values)
            
            # Write in TSV format: ID<tab>vector_values
            f.write(f"{vector_id}\t{vector_str}\n")
            
            if (i + 1) % 10000 == 0:
                print(f"Generated {i + 1} vectors...")
    
    print(f"Successfully created TSV file with {num_vectors} vectors")
    return tsv_file


def create_binary_file(output_dir: str, num_vectors: int, dimensions: int, data_type: str) -> str:
    """
    Create a binary file with synthetic zero vectors in SPTAG format.
    
    Format:
    <4 bytes int: num_vectors><4 bytes int: num_dimensions>
    <num_vectors * num_dimensions * sizeof(data_type) bytes: raw data>
    
    Args:
        output_dir: Output directory path
        num_vectors: Number of vectors to generate
        dimensions: Vector dimensionality
        data_type: Data type ('float' or 'int8')
    
    Returns:
        Path to created binary file
    """
    if data_type == 'float':
        extension = 'fbin'
        np_dtype = np.float32
        struct_format = 'f'
        dtype_size = 4
    elif data_type == 'int8':
        extension = 'i8bin'
        np_dtype = np.int8
        struct_format = 'b'
        dtype_size = 1
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    binary_file = os.path.join(output_dir, f"synthetic_{num_vectors}x{dimensions}_{data_type}.{extension}")
    
    print(f"Creating binary file: {binary_file}")
    print(f"Generating {num_vectors} vectors with {dimensions} dimensions of type {data_type}")
    
    with open(binary_file, 'wb') as f:
        # Write header: number of vectors and dimensions (both as 4-byte integers)
        f.write(struct.pack('i', num_vectors))
        f.write(struct.pack('i', dimensions))
        
        # Write vector data
        zero_vector = np.zeros(dimensions, dtype=np_dtype)
        
        for i in range(num_vectors):
            f.write(zero_vector.tobytes())
            
            if (i + 1) % 10000 == 0:
                print(f"Generated {i + 1} vectors...")
    
    print(f"Successfully created binary file with {num_vectors} vectors")
    print(f"File size: {os.path.getsize(binary_file)} bytes")
    print(f"Expected size: {8 + num_vectors * dimensions * dtype_size} bytes (header + data)")
    
    return binary_file


def verify_binary_file(binary_file: str, data_type: str, expected_vectors: int, expected_dimensions: int):
    """Verify the created binary file has correct format."""
    print(f"\nVerifying binary file: {binary_file}")
    
    if data_type == 'float':
        np_dtype = np.float32
        struct_format = 'f'
    else:  # int8
        np_dtype = np.int8
        struct_format = 'b'
    
    with open(binary_file, 'rb') as f:
        # Read header
        num_vectors = struct.unpack('i', f.read(4))[0]
        num_dimensions = struct.unpack('i', f.read(4))[0]
        
        print(f"Header - Vectors: {num_vectors}, Dimensions: {num_dimensions}")
        
        if num_vectors != expected_vectors or num_dimensions != expected_dimensions:
            print(f"ERROR: Header mismatch! Expected {expected_vectors}x{expected_dimensions}")
            return False
        
        # Read first vector to verify
        first_vector = np.frombuffer(f.read(num_dimensions * np_dtype().itemsize), dtype=np_dtype)
        print(f"First vector (first 10 values): {first_vector[:10]}")
        
        # Check if all values are zero
        if np.all(first_vector == 0):
            print("✓ First vector contains all zeros as expected")
        else:
            print("✗ First vector does not contain all zeros!")
            return False
    
    print("✓ Binary file verification passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic vector data for SPTAG testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate TSV file with 1000 float vectors of 128 dimensions
  python generate_synthetic_data.py -n 1000 -d 128 -t float
  
  # Generate binary file with 10000 int8 vectors of 256 dimensions
  python generate_synthetic_data.py -n 10000 -d 256 -t int8 --binary
  
  # Generate both TSV and binary files
  python generate_synthetic_data.py -n 5000 -d 128 -t int8 --binary
        """
    )
    
    parser.add_argument('-n', '--num-vectors', type=int, required=True,
                        help='Number of vectors to generate')
    parser.add_argument('-d', '--dimensions', type=int, required=True,
                        help='Vector dimensionality')
    parser.add_argument('-t', '--data-type', choices=['float', 'int8'], required=True,
                        help='Data type for vectors')
    parser.add_argument('--binary', action='store_true',
                        help='Generate binary format file (in addition to TSV)')
    parser.add_argument('-o', '--output-dir', type=str, default='synthetic_data',
                        help='Output directory (default: synthetic_data)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify generated binary files')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    print(f"Generating synthetic data:")
    print(f"  Vectors: {args.num_vectors}")
    print(f"  Dimensions: {args.dimensions}")
    print(f"  Data type: {args.data_type}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Binary format: {'Yes' if args.binary else 'No'}")
    print()
    
    # Always generate TSV file
    tsv_file = create_tsv_file(args.output_dir, args.num_vectors, args.dimensions, args.data_type)
    
    # Generate binary file if requested
    if args.binary:
        print()
        binary_file = create_binary_file(args.output_dir, args.num_vectors, args.dimensions, args.data_type)
        
        # Verify binary file if requested
        if args.verify:
            verify_binary_file(binary_file, args.data_type, args.num_vectors, args.dimensions)
    
    print(f"\n✓ Generation complete! Files created in: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
