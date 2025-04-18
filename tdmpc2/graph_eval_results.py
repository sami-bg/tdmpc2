import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # --- load data ---
    df = pd.read_csv('./evaluate_results.csv')
    tasks = df['task'].unique().tolist()
    n_tasks = len(tasks)

    # --- layout params ---
    max_cols = 3
    n_cols = min(n_tasks, max_cols)
    n_rows = math.ceil(n_tasks / n_cols)

    # bump up per‑panel size
    fig_width  = 7 * n_cols   # make it even wider
    fig_height = 5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, sharey=True,
                            figsize=(fig_width, fig_height))
    axes = axes.flatten() if n_tasks > 1 else [axes]

    # --- plot each task ---
    for ax, task in zip(axes, tasks):
        df_t = df[df['task'] == task]
        sns.barplot(
            x='ensemble_size',
            y='reward',
            hue='var_coeff',
            data=df_t,
            ci='sd',
            palette='colorblind',
            capsize=0.1,
            ax=ax
        )
        # remove the per‑axes legend seaborn drew
        if ax.legend_:
            ax.legend_.remove()

        ax.set_title(f'Task: {task}', pad=12)
        ax.set_xlabel('Ensemble Size')
        ax.set_ylabel('Mean Episode Reward (± Std Dev)')

    # kill any empty subplots
    for idx in range(len(tasks), n_rows * n_cols):
        fig.delaxes(axes[idx])

    # --- carve out the right 25% for legend ---
    # the rect arg is [left, bottom, right, top] in normalized figure coords
    plt.tight_layout(rect=[0, 0, 0.75, 1.0])

    # single global legend on the far right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title='Variance Coeff.',
        loc='upper right',
        bbox_to_anchor=(0.98, 0.95),
        frameon=True
    )

    # save with extra whitespace to ensure legend isn’t clipped
    fig.savefig('all_tasks_combined.png', dpi=300, bbox_inches='tight')
    print("Saved combined plot to all_tasks_combined.png")
