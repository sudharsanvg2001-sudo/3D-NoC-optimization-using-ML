import os

def generate_and_save_tsv_placements(R, C, density, max_solutions, filename="tsv_placements.txt"):
    """
    Generates valid TSV placement configurations using Bitmask Backtracking 
    and saves position numbers to a text file.
    
    Position indexing: 1-indexed (1 to R*C). 
    For a 4x4 grid:
      1  2  3  4
      5  6  7  8
      9 10 11 12
     13 14 15 16
    """
    K = int(R * C * density)  # Target number of TSVs
    
    # 1. Precompute valid single-row bitmasks (No horizontal neighbors)
    valid_masks = [m for m in range(1 << C) if (m & (m >> 1)) == 0]
    
    solutions = []

    def backtrack(row_idx, current_tsv_count, current_masks):
        nonlocal solutions
        
        # Stop searching if we reached target number of solutions
        if max_solutions is not None and len(solutions) >= max_solutions:
            return

        # Base Case: All rows assigned
        if row_idx == R:
            if current_tsv_count == K:
                solutions.append(list(current_masks))
            return

        # Pruning: If remaining rows cannot possibly reach K TSVs, abort branch
        remaining_rows = R - row_idx
        if current_tsv_count + (remaining_rows * (C // 2 + 1)) < K:
            return

        prev_mask = current_masks[-1] if current_masks else 0

        for mask in valid_masks:
            # Vertical neighbor constraint: No adjacent bits across rows
            if (mask & prev_mask) == 0:
                mask_cnt = bin(mask).count('1')
                if current_tsv_count + mask_cnt <= K:
                    current_masks.append(mask)
                    backtrack(row_idx + 1, current_tsv_count + mask_cnt, current_masks)
                    current_masks.pop()

    # Start recursion
    backtrack(0, 0, [])

    if not solutions:
        print(f"No valid placements found for {R}x{C} grid with {density*100}% target density.")
        return 0

    # 2. Convert bitmask configurations to 1-indexed Position Numbers
    formatted_solutions = []
    for sol_idx, masks in enumerate(solutions, start=1):
        positions = []
        for r_idx, mask in enumerate(masks):
            for c_idx in range(C):
                # Check if bit is set (reading left-to-right)
                if (mask >> (C - 1 - c_idx)) & 1:
                    pos_num = (r_idx * C) + c_idx + 1
                    positions.append(pos_num)
        formatted_solutions.append((sol_idx, positions))

    # 3. Save solutions to output file
    with open(filename, "w") as f:
        

        for sol_id, pos_list in formatted_solutions:
            pos_str = " ".join(map(str, pos_list))
            f.write(f"{pos_str}\n")

    print(f"Successfully generated {len(formatted_solutions)} solution(s).")
    print(f"Results saved to '{os.path.abspath(filename)}'.")
    return len(formatted_solutions)

# Execution Setup
R, C = 4, 4             # 8x8 Grid
DENSITY = 0.25          # 25% Density (16 TSVs)
MAX_SOLUTIONS = 5000   # Generate up to 500 unique valid layouts

generate_and_save_tsv_placements(
    R=R, 
    C=C, 
    density=DENSITY, 
    max_solutions=MAX_SOLUTIONS, 
    filename="tsv_placements_4x4.txt"
)