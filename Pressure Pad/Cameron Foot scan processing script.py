import csv
import matplotlib.pyplot as plt
import numpy as np

csv_filepath = r"C:\Users\camer\Documents\Imperial College London\MEng Design Engineering\Year 4\Master's Project\Validation\Uniform lattice pressure tests 18.5.25\graded lattice insole 5.csv"
scale_max = 120 # in kPa, for setting the colour scale of the plots
calibration_coefficient = 215

def map_raw_to_pressure(raw_value):
    return (raw_value*calibration_coefficient)/255

def map_raw_to_mass(raw_value):
    return (raw_value*calibration_coefficient*0.0702579)/(255*9.81)

def contact_area(foot_matrix):
    return np.count_nonzero(foot_matrix) * 0.702579

# --- LOAD AND PROCESS CSV DATA ---
def load_and_process_csv(filepath):
    """
    Loads a CSV file, filters headers/footers, finds a split point,
    splits the data, applies a mapping function to each part.

    Args:
        filepath (str): Path to the CSV file.
        mapping_func_callable (callable): Function to apply to each raw cell value.

    Returns:
        tuple: (processed_left_data, processed_right_data, split_description_string)
    """
    raw_numeric_rows = [] # Store rows of raw numbers after header/footer filtering
    skipped_row_count = 0
    total_mass = 0
    
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for r_idx, row_str_list in enumerate(reader):
                if not row_str_list:
                    skipped_row_count += 1
                    continue

                first_cell_raw = row_str_list[0]
                first_cell_stripped = first_cell_raw.strip()

                try:
                    int(first_cell_stripped) # Check if first cell is an integer
                    # This is a data row. Convert all cells to float.
                    numeric_row_segment = []
                    for c_idx, cell_val_str in enumerate(row_str_list):
                        try:
                            numeric_row_segment.append(float(cell_val_str.strip()))
                        except ValueError:
                            numeric_row_segment.append(np.nan) # Non-numeric in data row
                    raw_numeric_rows.append(numeric_row_segment)
                except ValueError:
                    # First cell not an int, skip this row (header/footer)
                    # print(f"Info: Skipping header/footer row (first cell '{first_cell_raw}' not int).")
                    skipped_row_count += 1
        
        if not raw_numeric_rows:
            return np.array([]), np.array([]), "No data rows found after header/footer filtering."

        # Ensure all data rows have same length for numpy conversion
        max_len = 0
        if raw_numeric_rows:
             max_len = max(len(r) for r in raw_numeric_rows if r) # Check if r is not None or empty
        
        padded_raw_numeric_rows = []
        for r in raw_numeric_rows:
            if r: # only process non-empty rows
                row_copy = list(r) # Make a copy to avoid modifying original list if it's referenced elsewhere
                while len(row_copy) < max_len:
                    row_copy.append(np.nan)
                padded_raw_numeric_rows.append(row_copy)
            
        if not padded_raw_numeric_rows:
             return np.array([]), np.array([]), "No valid data rows after padding."


        raw_data_np = np.array(padded_raw_numeric_rows, dtype=float)

    except FileNotFoundError:
        return np.array([]), np.array([]), f"Error: File not found at '{filepath}'"
    except Exception as e:
        return np.array([]), np.array([]), f"An error occurred during initial CSV processing: {e}"

    if raw_data_np.size == 0:
        return np.array([]), np.array([]), "No data to process after file reading."

    for row in raw_data_np:
        for value in row:
            total_mass += map_raw_to_mass(value)

    # Find the split column using the raw numerical data
    split_point, is_separator_zero_col, split_desc = find_split_column(raw_data_np)

    # Perform the split based on find_split_column's output
    if is_separator_zero_col:
        # split_point is the index of the zero column separator
        raw_left_np = raw_data_np[:, :split_point]
        raw_right_np = raw_data_np[:, split_point+1:]
    else:
        # split_point is the first column of the right part
        raw_left_np = raw_data_np[:, :split_point]
        raw_right_np = raw_data_np[:, split_point:]

    # Vectorize the mapping function for efficient application
    # This assumes map_raw_to_pressure is a scalar function (takes one raw_value)
    vectorized_mapper_pressure = np.vectorize(map_raw_to_pressure)

    processed_left_data = np.array([])
    processed_right_data = np.array([])

    if raw_left_np.size > 0:
        processed_left_data = vectorized_mapper_pressure(raw_left_np)
    elif raw_left_np.ndim == 2: # Preserve shape e.g. (N,0)
         processed_left_data = np.empty(raw_left_np.shape, dtype=float)

    if raw_right_np.size > 0:
        processed_right_data = vectorized_mapper_pressure(raw_right_np)
    elif raw_right_np.ndim == 2: # Preserve shape e.g. (N,0)
        processed_right_data = np.empty(raw_right_np.shape, dtype=float)
        
    return processed_left_data, processed_right_data, split_desc, total_mass


# --- VISUALISE MAPPED DATA ---
def visualise_pressure_map(pressure_data_orig, title="Calibrated Pressure Map", foot_label="", scale_max=120):
    """
    visualises the processed pressure data with enhancements:
    - Zeros are white.
    - Peak pressure is annotated.
    - Plot is cropped with padding.
    """
    if pressure_data_orig is None or pressure_data_orig.size == 0:
        if pressure_data_orig is not None and pressure_data_orig.ndim == 2 and pressure_data_orig.shape[1] == 0:
            print(f"Info: No data to visualise for {foot_label} (0 columns). Skipping plot.")
        else:
            print(f"Info: No data to visualise for {foot_label}. Data: {pressure_data_orig}")
        return

    pressure_data = pressure_data_orig.copy() # Work with a copy

    # --- Cropping Logic ---
    padding = 3
    # Valid data for cropping = non-NaN and non-zero
    valid_mask_for_crop = ~np.isnan(pressure_data) & (pressure_data != 0)
    
    crop_r_start, crop_c_start = 0, 0 # Offsets for peak annotation if cropped
    cropped_display_data = pressure_data # Default to original if no valid data to crop

    if np.any(valid_mask_for_crop):
        valid_rows = np.where(np.any(valid_mask_for_crop, axis=1))[0]
        valid_cols = np.where(np.any(valid_mask_for_crop, axis=0))[0]

        if valid_rows.size > 0 and valid_cols.size > 0:
            min_r, max_r = valid_rows.min(), valid_rows.max()
            min_c, max_c = valid_cols.min(), valid_cols.max()

            crop_r_start = max(0, min_r - padding)
            crop_r_end = min(pressure_data.shape[0], max_r + padding + 1)
            crop_c_start = max(0, min_c - padding)
            crop_c_end = min(pressure_data.shape[1], max_c + padding + 1)
            
            if crop_r_end > crop_r_start and crop_c_end > crop_c_start:
                 cropped_display_data = pressure_data[crop_r_start:crop_r_end, crop_c_start:crop_c_end]
            else: # Crop window is invalid, use original
                print(f"Info: Crop window for {foot_label} is invalid after padding. Using original data.")
                cropped_display_data = pressure_data
                crop_r_start, crop_c_start = 0,0 # Reset offsets
        else: # No valid rows/cols found (e.g. all NaNs or all zeros)
            cropped_display_data = pressure_data # Use original
            crop_r_start, crop_c_start = 0,0
    else: # No valid data points for cropping (all NaN or all zero)
        print(f"Info: No valid (non-zero, non-NaN) data for {foot_label} to determine crop. Displaying original.")
        cropped_display_data = pressure_data
        crop_r_start, crop_c_start = 0,0


    if cropped_display_data.size == 0:
        print(f"Info: Data for {foot_label} is empty after potential crop. Skipping plot.")
        return

        # Fixed color scale limits
    fixed_vmin = 0
    fixed_vmax = scale_max

    # Check for values greater than fixed_vmax (ignoring NaNs)
    max_val_in_cropped = np.nanmax(cropped_display_data)
    if max_val_in_cropped > fixed_vmax:
        raise ValueError(
            f"Error for '{foot_label}': Data contains value(s) greater than {fixed_vmax} "
            f"(max found: {max_val_in_cropped:.2f}). Plotting aborted. "
            "Adjust data or fixed_vmax."
        )
    
    # Check for values less than fixed_vmin (ignoring NaNs)
    min_val_in_cropped = np.nanmin(cropped_display_data)
    if min_val_in_cropped < fixed_vmin:
        # Allow exact zeros, as they will be colored white.
        # If there are non-zero values less than fixed_vmin (e.g. < 0), raise error.
        if np.any((cropped_display_data < fixed_vmin) & (cropped_display_data != 0) & ~np.isnan(cropped_display_data)):
             actual_min_negative = np.nanmin(cropped_display_data[cropped_display_data < fixed_vmin])
             raise ValueError(
                f"Error for '{foot_label}': Data contains value(s) less than {fixed_vmin} "
                f"(min found: {actual_min_negative:.2f}). Plotting aborted. "
                "Adjust data or fixed_vmin."
            )

    # --- Custom Colormap for White Zeros ---
    masked_values = np.ma.masked_where(cropped_display_data == 0, cropped_display_data)
    current_cmap = plt.cm.get_cmap('turbo').copy() 
    current_cmap.set_bad(color='white')

    plt.figure(figsize=(10, 8))
    
    img = plt.imshow(masked_values, cmap=current_cmap, interpolation='nearest', aspect='equal',
                     vmin=fixed_vmin, vmax=fixed_vmax)
    
    plt.colorbar(img, label='Pressure (kPa)')
    full_title = title
    if foot_label:
        full_title += f" - {foot_label}"
    plt.title(full_title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")

    # --- Peak Pressure Annotation ---
    try:
        # Find peak in the original (unmasked) cropped data
        peak_value = np.nanmax(cropped_display_data)
        if peak_value is not None and not np.isnan(peak_value) and peak_value > 0: # Annotate if peak is valid and positive
            # np.nanargmax flattens the array, so unravel
            peak_idx_flat = np.nanargmax(cropped_display_data)
            peak_r_cropped, peak_c_cropped = np.unravel_index(peak_idx_flat, cropped_display_data.shape)

            annotation_text = f"Peak: {peak_value:.2f} kPa"
            plt.annotate(annotation_text, xy=(peak_c_cropped, peak_r_cropped),
                         xytext=(peak_c_cropped + 2.0, peak_r_cropped + 2.0), # Offset text slightly
                         ha='left', va='bottom',
                         arrowprops=dict(facecolor='grey', edgecolor='grey', width=2, headwidth=2),
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8))
    except ValueError: # nanargmax raises ValueError if all are NaN
        print(f"Info: Could not find peak for {foot_label} (all NaN or empty after crop).")
    except Exception as e_annot:
        print(f"Warning: Error during peak annotation for {foot_label}: {e_annot}")


    # Adjust plot limits to the cropped data
    if cropped_display_data.shape[1] > 0:
        plt.xlim(-0.5, cropped_display_data.shape[1] - 0.5)
    if cropped_display_data.shape[0] > 0:
        plt.ylim(cropped_display_data.shape[0] - 0.5, -0.5)
    
    plt.tight_layout()
    plt.show()

# --- HELPER FUNCTION TO FIND SPLIT COLUMN ---
def find_split_column(raw_data_np):
    """
    Determines the column to split the data along for left/right foot separation.

    Args:
        raw_data_np (numpy.ndarray): 2D array of raw numerical data.

    Returns:
        tuple: (split_point, is_separator_zero_col, description_string)
               split_point: Index of the column.
               is_separator_zero_col: True if split_point is a zero column used as separator.
               description_string: String describing how the split was determined.
    """
    num_rows, num_cols = raw_data_np.shape
    split_description_parts = []

    if num_cols == 0:
        return 0, False, "No columns in data to split."
    if num_cols < 2:
        # Not enough columns for a meaningful split that results in two parts.
        # Split such that left has the one col, right is empty.
        split_description_parts.append(f"Fallback: Only {num_cols} column(s). Assigning all to left.")
        return num_cols, False, "".join(split_description_parts)


    # 1. Create a list of candidate columns (indices) that have zero readings in all cells.
    candidate_zero_cols = []
    for c in range(num_cols):
        if np.all(raw_data_np[:, c] == 0):
            candidate_zero_cols.append(c)
    split_description_parts.append(f"Found {len(candidate_zero_cols)} zero columns: {candidate_zero_cols}.")

    # 2. For each candidate column, check pressure distribution (30:70 ratio).
    valid_ratio_candidates = []
    if candidate_zero_cols:
        for c_idx in candidate_zero_cols:
            sum_left = np.sum(raw_data_np[:, :c_idx])
            sum_right = np.sum(raw_data_np[:, c_idx+1:]) # Exclude the zero column itself
            total_pressure_either_side = sum_left + sum_right

            is_balanced = False
            if total_pressure_either_side == 0: # Avoid division by zero; if all else is zero, it's balanced.
                is_balanced = True
            else:
                ratio_left = sum_left / total_pressure_either_side
                if 0.3 <= ratio_left <= 0.7:
                    is_balanced = True
            
            if is_balanced:
                valid_ratio_candidates.append(c_idx)
        split_description_parts.append(f"Zero columns with balanced (30:70) pressure: {valid_ratio_candidates}.")
    
    # 3. Look through remaining candidates and pick the most central one.
    #    It should have non-zero readings on both its left and right sides.
    if valid_ratio_candidates:
        potential_split_cols_info = []
        for c_idx in valid_ratio_candidates:
            l_edge = -1
            for lc in range(c_idx - 1, -1, -1):
                if np.any(raw_data_np[:, lc] != 0):
                    l_edge = lc
                    break
            
            r_edge = -1
            for rc in range(c_idx + 1, num_cols):
                if np.any(raw_data_np[:, rc] != 0):
                    r_edge = rc
                    break
            
            if l_edge != -1 and r_edge != -1: # Must have non-zero pressure on both sides of the zero gap
                potential_split_cols_info.append({'candidate': c_idx, 'L_edge': l_edge, 'R_edge': r_edge})

        if potential_split_cols_info:
            split_description_parts.append(f"Balanced zero cols with pressure on both sides: {[p['candidate'] for p in potential_split_cols_info]}.")
            
            best_candidate_info = None
            min_centrality_diff = float('inf')

            for p_info in potential_split_cols_info:
                c = p_info['candidate']
                l_e = p_info['L_edge'] # Rightmost pressure col left of c
                r_e = p_info['R_edge'] # Leftmost pressure col right of c
                
                # Minimize abs( (c - l_e) - (r_e - c) ) = abs(2*c - l_e - r_e)
                centrality_diff = abs(2 * c - l_e - r_e)
                
                if centrality_diff < min_centrality_diff:
                    min_centrality_diff = centrality_diff
                    best_candidate_info = p_info
                elif centrality_diff == min_centrality_diff:
                    # Tie-breaking: prefer candidate closer to the geometric center of the whole pad
                    if best_candidate_info is not None:
                        current_dist_to_pad_center = abs(c - (num_cols -1) / 2.0) # Center of pad (0 to num_cols-1)
                        best_dist_to_pad_center = abs(best_candidate_info['candidate'] - (num_cols-1) / 2.0)
                        if current_dist_to_pad_center < best_dist_to_pad_center:
                            best_candidate_info = p_info
            
            if best_candidate_info:
                chosen_zero_col = best_candidate_info['candidate']
                split_description_parts.append(f"Selected zero column {chosen_zero_col} as most central separator.")
                return chosen_zero_col, True, "\n".join(split_description_parts)

    # 4. Fallback: If steps above fail, use the middle column.
    #    split_point will be the first column of the right part.
    fallback_split_idx = num_cols // 2
    split_description_parts.append(f"Fallback: No suitable zero-column separator found. Splitting before column index {fallback_split_idx} (it becomes first of right part).")
    return fallback_split_idx, False, "\n".join(split_description_parts)

# --- MAIN SCRIPT EXECUTION ---
print(f"Attempting to load and process CSV file: {csv_filepath}")
left_foot_data, right_foot_data, split_description, total_mass = load_and_process_csv(csv_filepath)

print("--- Split Information ---")
print(split_description)
print("-------------------------\n")

if left_foot_data.size > 0 or right_foot_data.size > 0 :
    print(f"Processed Left Foot Data Shape: {left_foot_data.shape if left_foot_data is not None else 'None'}")
    if left_foot_data.size > 0 and not np.all(np.isnan(left_foot_data)):
            print(f"  Min/Max Calibrated (Left): {np.nanmin(left_foot_data):.2f} / {np.nanmax(left_foot_data):.2f}")
    
    print(f"Processed Right Foot Data Shape: {right_foot_data.shape if right_foot_data is not None else 'None'}")
    if right_foot_data.size > 0 and not np.all(np.isnan(right_foot_data)):
        print(f"  Min/Max Calibrated (Right): {np.nanmax(right_foot_data):.2f} / {np.nanmax(right_foot_data):.2f}")

    print("\nVisualising data...")
    visualise_pressure_map(left_foot_data, title="Calibrated Pressure Map", foot_label="Left Foot", scale_max=scale_max)
    visualise_pressure_map(right_foot_data, title="Calibrated Pressure Map", foot_label="Right Foot", scale_max=scale_max)
    print("Visualisation complete.")

    print(f"\n------ Left insole ------")
    print(f"Max pressure = {np.nanmax(left_foot_data):.2f} kPa")
    print(f"Contact area = {contact_area(left_foot_data):.2f} cm2")
    print(f"Pressure average over contact area = {(np.sum(left_foot_data)/np.count_nonzero(left_foot_data)):.2f} kPa")

    print(f"\n------ Right insole -----")
    print(f"Max pressure = {np.nanmax(right_foot_data):.2f} kPa")
    print(f"Contact area = {contact_area(right_foot_data):.2f} cm2")
    print(f"Pressure average over contact area = {(np.sum(right_foot_data)/np.count_nonzero(right_foot_data)):.2f} kPa")

    print(f"\n-------------------------------")
    print(f"Total mass of person = {total_mass:.2f} kg")
    print(f"-------------------------------")

else:
    print("\nScript finished: No data was processed from either foot or an error occurred.")