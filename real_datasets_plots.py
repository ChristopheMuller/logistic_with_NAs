
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

#%%

def plot_real_datasets_grid(df, datasets, metrics, methods, output_path=None):
    df_filtered = df[
        (df["Dataset"].isin(datasets)) &
        (df["Metric"].isin(metrics)) &
        (df["Method"].isin(methods))
    ].copy()

    df_agg = df_filtered.groupby(["Dataset", "Metric", "Method"])["Value"].agg(["mean", "std", "count"]).reset_index()
    df_agg["se"] = df_agg["std"] / np.sqrt(df_agg["count"])

    num_rows = len(datasets)
    num_cols = len(metrics)
    
    fig_height = 2.5 * num_rows
    fig_width = 3.5 * num_cols + 2 

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(num_rows, num_cols + 1, width_ratios=[1] * num_cols + [0.3])

    unique_methods = sorted(methods)
    colors = sns.color_palette("husl", len(unique_methods))
    method_colors = dict(zip(unique_methods, colors))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h'] * 3
    method_markers = dict(zip(unique_methods, markers[:len(unique_methods)]))

    for r_idx, d_name in enumerate(datasets):
        
        row_label_ax = fig.add_subplot(gs[r_idx, 0])
        row_label_ax.text(-0.25, 0.5, d_name, transform=row_label_ax.transAxes, 
                         fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
        row_label_ax.axis('off')

        for c_idx, m_name in enumerate(metrics):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            
            data_subset = df_agg[
                (df_agg["Dataset"] == d_name) & 
                (df_agg["Metric"] == m_name)
            ]

            if data_subset.empty:
                ax.axis('off')
                continue

            for method in methods:
                method_data = data_subset[data_subset["Method"] == method]
                
                if not method_data.empty:
                    mean_val = method_data["mean"].values[0]
                    se_val = method_data["se"].values[0]
                    
                    ax.errorbar(
                        x=method, 
                        y=mean_val, 
                        yerr=se_val, 
                        fmt='none', 
                        ecolor=method_colors[method], 
                        capsize=3, 
                        alpha=0.7
                    )
                    
                    ax.plot(
                        method, 
                        mean_val, 
                        marker=method_markers[method], 
                        color=method_colors[method], 
                        markersize=6, 
                        linestyle='None',
                        label=method if (r_idx == 0 and c_idx == 0) else ""
                    )

            if r_idx == 0:
                ax.set_title(m_name, fontsize=13, pad=10)

            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            y_min = data_subset["mean"].min()
            y_max = data_subset["mean"].max()
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 0.1
            ax.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.2)

    legend_ax = fig.add_subplot(gs[:, num_cols])
    legend_ax.axis('off')
    
    handles = []
    labels = []
    for method in unique_methods:
        h = plt.Line2D([0], [0], marker=method_markers[method], color=method_colors[method], 
                      linestyle='None', markersize=8)
        handles.append(h)
        labels.append(method)

    legend_ax.legend(handles, labels, loc='center left', title="Methods", frameon=False, fontsize=11)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    plt.show()

#%% Example usage
if __name__ == "__main__":
    try:
        df = pd.read_csv("real_datasets_results/real_datasets_metrics_detailed.csv")

        selected_datasets = sorted(df["Dataset"].unique().tolist())
        selected_metrics = ["AUC", "Brier", "Misclassification", "Calibration"]
        selected_methods = sorted(df["Method"].unique().tolist()) 

        plot_real_datasets_grid(df, selected_datasets, selected_metrics, selected_methods)
        
    except FileNotFoundError:
        print("CSV file not found. Please ensure the path is correct.")
# %%
