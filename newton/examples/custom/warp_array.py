"""
Complete Guide to Printing Warp Arrays

Warp arrays live on GPU/CPU and need to be converted to NumPy for inspection.
"""

import warp as wp
import numpy as np

# ============================================================================
# BASIC PRINTING
# ============================================================================

def basic_printing():
    """Basic ways to print warp arrays"""
    
    # Create a simple array
    arr = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    
    print("=== BASIC PRINTING ===\n")
    
    # Method 1: Convert to numpy first (RECOMMENDED)
    print("Method 1 - Convert to numpy:")
    print(arr.numpy())
    print()
    
    # Method 2: Direct print (shows metadata, not values)
    print("Method 2 - Direct print (not useful):")
    print(arr)  # Shows: <warp.array object at 0x...>
    print()
    
    # Method 3: Print array info
    print("Method 3 - Array info:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Device: {arr.device}")
    print(f"  Size: {arr.size}")
    print()


# ============================================================================
# PRINTING SPECIFIC ELEMENTS
# ============================================================================

def print_elements():
    """Print specific elements or slices"""
    
    arr = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    
    print("=== PRINTING ELEMENTS ===\n")
    
    # Single element
    print(f"First element: {arr.numpy()[0]}")
    print(f"Last element: {arr.numpy()[-1]}")
    print()
    
    # Slice
    print(f"First 3 elements: {arr.numpy()[:3]}")
    print()
    
    # Specific indices
    indices = [0, 2, 4]
    print(f"Elements at indices {indices}: {arr.numpy()[indices]}")
    print()


# ============================================================================
# PRINTING VECTOR ARRAYS
# ============================================================================

def print_vectors():
    """Print arrays of vec3, vec2, etc."""
    
    print("=== PRINTING VECTORS ===\n")
    
    # Create vec3 array
    vectors = wp.array([
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
        wp.vec3(7.0, 8.0, 9.0)
    ], dtype=wp.vec3)
    
    # Convert to numpy
    vec_numpy = vectors.numpy()
    
    print("All vectors:")
    print(vec_numpy)
    print()
    
    print("First vector:")
    print(vec_numpy[0])
    print()
    
    # Access components
    print("X components of all vectors:")
    print(vec_numpy[:, 0])  # Note: vec3 becomes shape (N, 3) in numpy
    print()


# ============================================================================
# PRINTING 2D ARRAYS
# ============================================================================

def print_2d_arrays():
    """Print 2D arrays and matrices"""
    
    print("=== PRINTING 2D ARRAYS ===\n")
    
    # Create 2D array
    arr_2d = wp.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=int)
    
    print("Full 2D array:")
    print(arr_2d.numpy())
    print()
    
    print("First row:")
    print(arr_2d.numpy()[0, :])
    print()
    
    print("Second column:")
    print(arr_2d.numpy()[:, 1])
    print()
    
    print(f"Element at [1, 2]: {arr_2d.numpy()[1, 2]}")
    print()


# ============================================================================
# PRINTING LARGE ARRAYS (WITH FORMATTING)
# ============================================================================

def print_large_arrays():
    """Print large arrays with nice formatting"""
    
    print("=== PRINTING LARGE ARRAYS ===\n")
    
    # Create large array
    large_arr = wp.array(np.random.randn(100), dtype=float)
    
    # Method 1: Print first N elements
    N = 5
    print(f"First {N} elements:")
    print(large_arr.numpy()[:N])
    print()
    
    # Method 2: Print with numpy formatting
    np.set_printoptions(precision=3, suppress=True)
    print("First 10 elements (formatted):")
    print(large_arr.numpy()[:10])
    print()
    
    # Method 3: Print summary statistics
    print("Array statistics:")
    print(f"  Min: {large_arr.numpy().min():.4f}")
    print(f"  Max: {large_arr.numpy().max():.4f}")
    print(f"  Mean: {large_arr.numpy().mean():.4f}")
    print(f"  Std: {large_arr.numpy().std():.4f}")
    print()


# ============================================================================
# PRINTING CONTACT ARRAYS (NEWTON SPECIFIC)
# ============================================================================

def print_contact_arrays():
    """Example of printing Newton contact arrays"""
    
    print("=== PRINTING CONTACT ARRAYS ===\n")
    
    # Simulate Newton contact structure
    contact_count = wp.array([5], dtype=wp.int32)
    contact_particle = wp.array([0, 1, 2, 3, 4, -1, -1], dtype=int)
    contact_normal = wp.array([
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 0.0),
    ], dtype=wp.vec3)
    
    # Get actual count
    # count = int(contact_count.numpy()[0])
    count = int(contact_count.numpy()[0])
    print(f"Number of contacts: {count}")
    print()
    
    # Print valid contacts only
    print("Contact particles:")
    print(contact_particle.numpy()[:count])
    print()
    
    print("Contact normals:")
    normals = contact_normal.numpy()[:count]
    for i, normal in enumerate(normals):
        print(f"  Contact {i}: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
    print()


# ============================================================================
# FORMATTED PRINTING UTILITIES
# ============================================================================

def pretty_print_array(arr: wp.array, name: str = "Array", max_items: int = 10):
    """Pretty print a warp array with formatting"""
    
    print(f"\n{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Device: {arr.device}")
    
    numpy_arr = arr.numpy()
    
    if arr.size <= max_items:
        print(f"  Values: {numpy_arr}")
    else:
        print(f"  First {max_items} values: {numpy_arr.flatten()[:max_items]}")
        print(f"  ... ({arr.size - max_items} more elements)")


def print_nonzero(arr: wp.array, name: str = "Array"):
    """Print only non-zero elements"""
    
    numpy_arr = arr.numpy()
    nonzero_indices = np.nonzero(numpy_arr)[0]
    
    print(f"\n{name} - Non-zero elements:")
    if len(nonzero_indices) == 0:
        print("  (none)")
    else:
        for idx in nonzero_indices[:20]:  # Limit to first 20
            print(f"  [{idx}] = {numpy_arr[idx]}")
        if len(nonzero_indices) > 20:
            print(f"  ... ({len(nonzero_indices) - 20} more)")


# ============================================================================
# PRACTICAL NEWTON EXAMPLE
# ============================================================================

def newton_contact_printing_example():
    """Realistic example of printing Newton simulation data"""
    
    print("\n" + "="*60)
    print("NEWTON SIMULATION DATA PRINTING EXAMPLE")
    print("="*60 + "\n")
    
    # Simulate Newton data structures
    particle_count = 27  # 3x3x3 grid
    particle_q = wp.array(np.random.randn(particle_count, 3), dtype=wp.vec3)
    particle_qd = wp.array(np.random.randn(particle_count, 3) * 0.1, dtype=wp.vec3)
    particle_f = wp.array(np.random.randn(particle_count, 3) * 100, dtype=wp.vec3)
    
    # Contact data
    soft_contact_count = wp.array([3], dtype=wp.int32)
    soft_contact_particle = wp.array([5, 8, 12, -1, -1], dtype=int)
    soft_contact_normal = wp.array([
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.1, 0.0, 0.99),
        wp.vec3(-0.1, 0.0, 0.99),
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 0.0),
    ], dtype=wp.vec3)
    
    # Print particle info
    print("PARTICLE DATA:")
    print(f"  Total particles: {particle_count}")
    print(f"  First 3 positions:")
    for i in range(3):
        pos = particle_q.numpy()[i]
        print(f"    Particle {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    print()
    
    # Print contact info
    count = int(soft_contact_count.numpy()[0])
    print(f"CONTACT DATA:")
    print(f"  Number of contacts: {count}")
    
    if count > 0:
        particles = soft_contact_particle.numpy()[:count]
        normals = soft_contact_normal.numpy()[:count]
        
        print(f"  Contacting particles: {particles}")
        print(f"  Contact details:")
        for i in range(count):
            n = normals[i]
            print(f"    Contact {i}: particle {particles[i]}, "
                  f"normal ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})")
    print()
    
    # Print force statistics
    forces = particle_f.numpy()
    force_magnitudes = np.linalg.norm(forces, axis=1)
    
    print("FORCE DATA:")
    print(f"  Max force magnitude: {force_magnitudes.max():.2f} N")
    print(f"  Mean force magnitude: {force_magnitudes.mean():.2f} N")
    print(f"  Particles with force > 50N: {np.sum(force_magnitudes > 50)}")
    
    # Find particles with highest forces
    top_indices = np.argsort(force_magnitudes)[-3:][::-1]
    print(f"  Top 3 particles by force:")
    for idx in top_indices:
        f = forces[idx]
        mag = force_magnitudes[idx]
        print(f"    Particle {idx}: {mag:.2f}N, force=({f[0]:.1f}, {f[1]:.1f}, {f[2]:.1f})")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    wp.init()
    
    # basic_printing()
    # print_elements()
    # print_vectors()
    # print_2d_arrays()
    # print_large_arrays()
    print_contact_arrays()
    
    # Utility examples
    test_arr = wp.array([0, 0, 5, 0, 0, 10, 0, 15], dtype=float)
    pretty_print_array(test_arr, "Test Array", max_items=5)
    print_nonzero(test_arr, "Test Array")
    
    # Newton example
    newton_contact_printing_example()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAY: Always use .numpy() to convert Warp arrays!")
    print("="*60)
