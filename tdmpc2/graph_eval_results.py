import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate bar plots for specific experimental results.')
    parser.add_argument('--csv_path', type=str, default='./evaluate_results.csv',
                        help='Path to the input CSV file (default: ./evaluate_results.csv)')
    parser.add_argument('--task', type=str, required=True,
                        help='The specific task to plot (e.g., humanoid-run, dog-run).')
    parser.add_argument('--aggregation', type=str, required=True,
                        help='The aggregation method to plot (e.g., mean, max, min).')
    parser.add_argument('--horizon_eval', type=int, required=True,
                        help='The evaluation horizon to plot (e.g., 3, 5).')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the plot (default: current directory).')
    return parser.parse_args()

def plot_results(df_filtered, task, aggregation, horizon_eval, output_dir):
    """Generates and saves a bar plot for the filtered data."""
    if df_filtered.empty:
        print(f"Error: No data found for task='{task}', aggregation='{aggregation}', horizon_eval={horizon_eval}.")
        print("Please check your input arguments and the CSV file.")
        sys.exit(1) # Exit if no data to plot

    # --- Create Plot ---
    # Using seaborn-v0_8-colorblind style for overall plot aesthetics
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(10, 6)) # Single plot figure

    # --- Generate Bar Plot ---
    # Seaborn automatically calculates the mean of 'reward' for each combination
    # of 'ensemble_size' and 'var_coeff'.
    # The errorbar='sd' argument calculates the standard deviation across the different
    # 'seed' values for each bar group and displays it as error bars.
    sns.barplot(
        x='ensemble_size',
        y='reward',
        hue='var_coeff',
        data=df_filtered,
        errorbar='sd', # Use standard deviation for error bars based on 'seed'
        palette='tab10', # Changed palette to 'tab10' for better visibility
        capsize=0.1,
        ax=ax
    )

    # --- Customize Plot ---
    plot_title = f'Task: {task} | Aggregation: {aggregation} | Horizon: {horizon_eval}'
    ax.set_title(plot_title, pad=20, fontsize=14)
    ax.set_xlabel('Ensemble Size', fontsize=12)
    ax.set_ylabel('Mean Episode Reward (± Std Dev over Seeds)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # --- Add Legend ---
    # Place legend outside the plot area to avoid overlap
    handles, labels = ax.get_legend_handles_labels()
    # Sort legend items numerically by variance coefficient (label) if possible
    try:
        # Attempt to sort labels numerically
        sorted_indices = sorted(range(len(labels)), key=lambda k: float(labels[k]))
        handles = [handles[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    except ValueError:
        # If labels are not purely numeric, keep original order
        pass

    fig.legend(
        handles, labels,
        title='Variance Coeff.',
        loc='center right',
        bbox_to_anchor=(1.15, 0.5), # Adjust position as needed
        frameon=True,
        title_fontsize='12',
        fontsize='10'
    )
    # Remove the automatic legend inside the plot if seaborn added one
    if ax.legend_:
        ax.legend_.remove()

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust right boundary to make space for legend

    # --- Save Plot ---
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a descriptive filename
    filename = f"plot_task_{task}_agg_{aggregation}_horizon_{horizon_eval}.png"
    filepath = os.path.join(output_dir, filename)

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
        # Ensure 'var_coeff' is treated as a categorical variable for consistent hue mapping
        if 'var_coeff' in df.columns:
             df['var_coeff'] = df['var_coeff'].astype(str)

    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        sys.exit(1)

    # --- Filter Data ---
    # Convert horizon_eval argument to the same type as the column for comparison
    df_filtered = df[
        (df['task'] == args.task) &
        (df['aggregation'] == args.aggregation) &
        (df['horizon_eval'] == args.horizon_eval) # Direct comparison assuming column is numeric
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    # --- Plot Filtered Data ---
    plot_results(df_filtered, args.task, args.aggregation, args.horizon_eval, args.output_dir)

if __name__ == '__main__':
    main()
