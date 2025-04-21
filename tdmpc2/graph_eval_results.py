import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import math
from itertools import product

# Define columns that can be used for plotting dimensions
PLOT_DIMENSION_COLS = ['ensemble_size', 'horizon_eval', 'var_coeff', 'aggregation']
# Define columns that are fixed roles
FIXED_COLS = ['task', 'reward', 'seed', 'success']

def parse_arguments():
    """Parses command-line arguments for generalized plotting."""
    parser = argparse.ArgumentParser(
        description='Generate flexible grid plots for experimental results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--csv_path', type=str, default='./evaluate_results.csv',
                        help='Path to the input CSV file.')
    parser.add_argument('--task', type=str, required=True,
                        help='The specific task to filter data for.')
    parser.add_argument('--row-by', type=str, default=None, choices=PLOT_DIMENSION_COLS + [None],
                        help='Column name to use for subplot rows.')
    parser.add_argument('--col-by', type=str, default=None, choices=PLOT_DIMENSION_COLS + [None],
                        help='Column name to use for subplot columns.')
    parser.add_argument('--x-by', type=str, required=True, choices=PLOT_DIMENSION_COLS,
                        help='Column name to use for the x-axis in subplots.')
    parser.add_argument('--hue-by', type=str, required=True, choices=PLOT_DIMENSION_COLS,
                        help='Column name to use for bar groups (hue) in subplots.')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the plot.')
    parser.add_argument('--output_filename', type=str, default=None,
                        help='Custom filename for the output plot. If None, a name is generated.')
    parser.add_argument('--palette', type=str, default='tab10',
                        help='Color palette for seaborn plots (e.g., tab10, viridis, colorblind).')
    parser.add_argument('--share-y', action='store_true', default=True,
                        help='Share Y-axis across all subplots.')
    parser.add_argument('--no-share-y', action='store_false', dest='share_y',
                        help='Do not share Y-axis across subplots.')


    args = parser.parse_args()

    # --- Validate Arguments ---
    used_dims = {args.row_by, args.col_by, args.x_by, args.hue_by}
    used_dims.discard(None) # Remove None if present

    if len(used_dims) != len([d for d in [args.row_by, args.col_by, args.x_by, args.hue_by] if d is not None]):
        print("Error: The same column cannot be used for multiple plot dimensions (row, col, x, hue).")
        sys.exit(1)

    # Ensure required args are provided (already handled by required=True)
    if not args.x_by or not args.hue_by:
         print("Error: --x-by and --hue-by arguments are required.")
         sys.exit(1)


    return args

def get_sorted_unique_values(df, column_name):
    """Gets unique values from a column and sorts them numerically if possible,
       falling back to lexicographical sorting."""
    if column_name is None or column_name not in df.columns:
        return [None] # Return a list containing None for iteration

    # Get unique values, ensuring NaNs are handled if they exist in the original col
    unique_vals = df[column_name].dropna().unique()
    if len(unique_vals) == 0:
        return [] # No valid unique values found

    num_col_name = f"{column_name}_num"

    # Attempt numeric sort if the _num column exists
    if num_col_name in df.columns:
        # Create a map from the original value (string) to the numeric value
        # Important: Use the original df, not just unique_vals, to build the map correctly
        numeric_map = df.set_index(column_name)[num_col_name].to_dict()
        try:
            # Filter unique values to those that have a valid numeric counterpart
            valid_numeric_pairs = []
            for v in unique_vals:
                num_v = numeric_map.get(v)
                if pd.notna(num_v):
                    valid_numeric_pairs.append((v, num_v))

            # If we found numeric counterparts, sort by them
            if valid_numeric_pairs:
                valid_numeric_pairs.sort(key=lambda item: item[1]) # Sort by numeric value
                sorted_vals = [item[0] for item in valid_numeric_pairs] # Extract sorted original values
                return sorted_vals
            else:
                # Fallback: No valid numeric counterparts found, sort original values lexicographically
                print(f"Warning: Could not sort '{column_name}' numerically. Using lexicographical sort.")
                return sorted(list(unique_vals))

        except Exception as e:
             # Fallback on any exception during numeric sort attempt
             print(f"Warning: Exception during numeric sort for '{column_name}': {e}. Using lexicographical sort.")
             return sorted(list(unique_vals)) # Ensure unique_vals is a list
    else:
        # Fallback: _num column doesn't exist, sort original values lexicographically
        return sorted(list(unique_vals)) # Ensure unique_vals is a list


def plot_grid(df, args):
    """Generates and saves a generalized grid plot based on arguments."""

    row_col = args.row_by
    col_col = args.col_by
    x_col = args.x_by
    hue_col = args.hue_by
    y_col = 'reward' # Fixed Y axis

    # --- Get Unique Values for Rows and Columns ---
    unique_rows = get_sorted_unique_values(df, row_col)
    unique_cols = get_sorted_unique_values(df, col_col)

    n_rows = len(unique_rows)
    n_cols = len(unique_cols)

    if n_rows == 0 or n_cols == 0:
        print("Error: No data available for the specified row/column dimensions.")
        sys.exit(1)

    # --- Create Plot ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols + 2, 5 * n_rows + 1), # Adjust size based on grid, add space for legend/titles
        sharey=args.share_y,
        squeeze=False # Ensure axes is always 2D
    )

    # --- Iterate Through Grid and Plot ---
    plot_generated = False
    first_valid_ax = None # To get legend handles later

    for r_idx, row_val in enumerate(unique_rows):
        for c_idx, col_val in enumerate(unique_cols):
            ax = axes[r_idx, c_idx]
            df_cell = df.copy() # Start with the full filtered data for the task

            # Filter for current row value
            if row_col is not None:
                df_cell = df_cell[df_cell[row_col] == row_val]

            # Filter for current column value
            if col_col is not None:
                df_cell = df_cell[df_cell[col_col] == col_val]

            # --- Plot Cell ---
            if df_cell.empty or df_cell[y_col].isnull().all():
                # Handle empty cells
                title = []
                if row_col: title.append(f"{row_col}={row_val}")
                if col_col: title.append(f"{col_col}={col_val}")
                ax.set_title(" | ".join(title) + " (No Data)", fontsize=10, pad=5)
                ax.axis('off')
                continue # Skip plotting for this empty cell

            plot_generated = True
            if first_valid_ax is None:
                first_valid_ax = ax

            # Sort data by numeric x-axis value before plotting
            x_num_col = f"{x_col}_num"
            if x_num_col in df_cell.columns:
                 try:
                     df_cell.sort_values(x_num_col, inplace=True)
                 except Exception: # Fallback sorting
                     df_cell.sort_values(x_col, inplace=True)
            else:
                 df_cell.sort_values(x_col, inplace=True)


            sns.barplot(
                x=x_col,
                y=y_col,
                hue=hue_col,
                data=df_cell,
                errorbar='sd', # Std dev over seeds
                palette=args.palette,
                capsize=0.1,
                ax=ax
            )

            # --- Customize Subplot ---
            title = []
            if row_col: title.append(f"{row_col}={row_val}")
            if col_col: title.append(f"{col_col}={col_val}")
            ax.set_title(" | ".join(title) if title else "Plot", fontsize=11, pad=8)

            ax.set_xlabel(x_col, fontsize=10)
            ax.tick_params(axis='x', rotation=30, labelsize=9) # Rotate x-ticks slightly
            ax.tick_params(axis='y', labelsize=9)


            # Set Y label only on the first column
            if c_idx == 0:
                ax.set_ylabel(f"{y_col} (± Std Dev)", fontsize=10)
            else:
                ax.set_ylabel("")
                if args.share_y: # Remove y-tick labels if sharing y axis and not first col
                     ax.tick_params(axis='y', labelleft=False)


            # Remove individual legends
            if ax.legend_:
                ax.legend_.remove()

    if not plot_generated:
        print("Error: No data found to generate any plots after filtering.")
        plt.close(fig)
        sys.exit(1)

    # --- Add Overall Figure Title ---
    fig.suptitle(f'Task: {args.task}', fontsize=16, y=0.99) # Adjust y to fit

    # --- Add Single Global Legend ---
    if first_valid_ax:
        handles, labels = first_valid_ax.get_legend_handles_labels()
        # Sort legend items numerically by hue value (label) if possible
        hue_num_col = f"{hue_col}_num"
        if hue_num_col in df.columns:
             try:
                 # Create a map from string label to numeric value for sorting
                 label_to_num_map = df.set_index(hue_col)[hue_num_col].to_dict()
                 valid_labels = [(l, label_to_num_map.get(l)) for l in labels if pd.notna(label_to_num_map.get(l))]
                 valid_labels.sort(key=lambda item: item[1]) # Sort by numeric value
                 sorted_indices = [labels.index(l[0]) for l in valid_labels] # Get original indices
                 handles = [handles[i] for i in sorted_indices]
                 labels = [labels[i] for i in sorted_indices]
             except Exception as e:
                 print(f"Warning: Could not sort legend numerically - {e}")
                 pass # Keep original order if sorting fails

        fig.legend(
            handles, labels,
            title=hue_col, # Legend title is the hue column name
            loc='center right',
            bbox_to_anchor=(1.0 + 0.02 * n_cols, 0.5), # Adjust position based on num cols
            frameon=True,
            title_fontsize='12',
            fontsize='10'
        )

    # Adjust layout (decrease right boundary to make space for legend)
    plt.tight_layout(rect=[0, 0, 0.98 - 0.03 * n_cols, 0.95]) # Adjust rect [left, bottom, right, top]

    # --- Save Plot ---
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_filename:
        filename = args.output_filename
    else:
        # Generate filename
        dims = f"row_{args.row_by}_col_{args.col_by}_x_{args.x_by}_hue_{args.hue_by}".replace("__", "_")
        filename = f"plot_task_{args.task}_{dims}.png"
    filepath = os.path.join(args.output_dir, filename)

    try:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        sys.exit(1)

    plt.close(fig) # Close the figure to free memory


def main():
    """Main function to load data, filter, and plot."""
    args = parse_arguments()

    # --- Load Data ---
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # --- Validate Columns Exist ---
    all_used_cols = {args.row_by, args.col_by, args.x_by, args.hue_by, 'task', 'reward', 'seed'}
    all_used_cols.discard(None)
    missing_cols = all_used_cols - set(df.columns)
    if missing_cols:
        print(f"Error: The following specified columns are missing from the CSV: {missing_cols}")
        sys.exit(1)


    # --- Data Type Handling ---
    # Convert potential dimension columns to numeric (for sorting) and string (for plotting)
    for col in PLOT_DIMENSION_COLS:
        if col in df.columns:
            num_col = f"{col}_num"
            # Create numeric column for sorting, coercing errors
            df[num_col] = pd.to_numeric(df[col], errors='coerce')
            # Convert original column to string for categorical plotting/grouping
            # Handle potential NaN values before converting to string if necessary
            if df[col].isnull().any():
                 df[col] = df[col].fillna('NaN').astype(str)
            else:
                 df[col] = df[col].astype(str)


    # --- Filter Data by Task ---
    df_filtered = df[df['task'] == args.task].copy()

    if df_filtered.empty:
        print(f"Error: No data found for task='{args.task}'.")
        sys.exit(1)

    # --- Plot Filtered Data ---
    plot_grid(df_filtered, args)

if __name__ == '__main__':
    main()
