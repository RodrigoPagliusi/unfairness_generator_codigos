# visualizer.py

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


class Visualizer:

    def __init__(self, config_visual):

        self.results_directory = "results/"
        self.plots_directory = config_visual['visualization']['plots_directory']
        os.makedirs(self.plots_directory, exist_ok=True)
        self.results_df = None
        self.differences_df = None
        self.cumulative_differences_df = None
        self.dataset = config_visual['visualization']['dataset']
        self.plots_list = config_visual['visualization']['plots']
        self.threshold_values = config_visual['visualization']['thresholds']
        self.threshold_values = [float(t) for t in self.threshold_values]
        self.id_columns = [
            'dataset', 'processing_stage', 'estimator_model', 'classification_model',
            'tune_hyperparameters', 'num_trials', 'optimization_metric', 'flip_method'
        ]
        self.multiple_lines = len(self.plots_list) > 1

    def load_evaluation_results(self):

        results_filename = os.path.join(
            self.results_directory, 'evaluation_results.xlsx')
        if os.path.isfile(results_filename):
            self.results_df = pd.read_excel(
                results_filename,
                sheet_name='Results')
            logging.info(f"Evaluation results loaded from {results_filename}")

            self.results_df['Threshold'] = self.results_df['Threshold'].astype(
                str)
            self.results_df = self.results_df[~self.results_df['Threshold'].str.contains(
                '-')]
            self.results_df['Threshold'] = self.results_df['Threshold'].astype(
                float)

            self.differences_df = pd.read_excel(
                results_filename,
                sheet_name='Differences')
            logging.info(f"Differences data loaded from {results_filename}")

            self.differences_df['Threshold'] = self.differences_df['Threshold'].astype(
                str)
            self.differences_df[['Current Threshold', 'Previous Threshold']] = self.differences_df['Threshold'].str.extract(
                r'([0-9.]+)\s*-\s*([0-9.]+)').astype(float)

            self.cumulative_differences_df = pd.read_excel(
                results_filename,
                sheet_name='Cumulative Differences')
            logging.info(f"Cumulative differences data loaded from {
                         results_filename}")

            self.cumulative_differences_df['Threshold'] = self.cumulative_differences_df['Threshold'].astype(
                str)
            self.cumulative_differences_df[['Current Threshold', 'Previous Threshold']] = self.cumulative_differences_df['Threshold'].str.extract(
                r'([0-9.]+)\s*-\s*([0-9.]+)').astype(float)

        else:
            logging.error(f"Results file not found at {results_filename}")
            raise FileNotFoundError(
                f"Results file not found at {results_filename}")

    def filter_results_df(self, df, line):

        filtered_df = df[
            (df['dataset'] == self.dataset) &
            (df['flip_method'] == line['flip_method']) &
            (df['estimator_model'] == line['estimator_model']) &
            (df['classification_model'] == line['classification_model'])
        ].copy()

        if filtered_df.empty:
            logging.warning(f"No data found for line: {line}")
            return None

        if 'Threshold' in filtered_df.columns and filtered_df['Threshold'].dtype == float:
            filtered_df = filtered_df.sort_values('Threshold')
        elif 'Current Threshold' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('Current Threshold')
        return filtered_df

    def plot_metrics(self):

        if self.results_df is None or self.differences_df is None or self.cumulative_differences_df is None:
            logging.error(
                "Evaluation results or differences not loaded. Call load_evaluation_results() first.")
            raise ValueError("Evaluation results or differences not loaded.")

        performance_metrics = ['Accuracy', 'F1 Score',
                               'Matthews Correlation Coefficient']
        fairness_metrics = ['Statistical Parity Difference',
                            'Equal Opportunity Difference', 'Equalized Odds Difference']
        all_metrics = performance_metrics + fairness_metrics

        global_max_diff = self._compute_global_max(
            self.differences_df, self.cumulative_differences_df, all_metrics)

        self._plot_metric_set(self.results_df, all_metrics,
                              'Threshold', 'Results', 'A', plot_number=7)
        self._plot_metric_set(self.differences_df, all_metrics, 'Current Threshold',
                              'Differences', 'B', plot_number=7, use_combined_plot=True)

    def _compute_global_max(self, differences_df, cumulative_df, metrics):

        max_vals = []
        for df in [differences_df, cumulative_df]:
            for metric in metrics:
                if metric in df.columns:
                    max_val = df[metric].abs().max()
                    if pd.notnull(max_val):
                        max_vals.append(max_val)
        if max_vals:
            limit = max(max_vals)
            limit = np.ceil(limit / 0.05) * 0.05
            return limit
        else:
            return None

    def _plot_metric_set(self, df, metrics, threshold_col, data_label, sub_label, plot_number, use_combined_plot=False):

        num_metrics = len(metrics)
        num_rows = (num_metrics + 2) // 3
        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.6, bottom=-0.15)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            ax.set_xlabel('Threshold')
            ax.set_ylabel(metric)
            y_values = []

            if threshold_col == 'Threshold' and not use_combined_plot:
                ax.set_xticks(self.threshold_values)

            for line in self.plots_list:
                filtered_df = self.filter_results_df(df, line)
                if filtered_df is not None:
                    label = f"{line['estimator_model']}_{
                        line['classification_model']}_{line['flip_method']}"
                    if use_combined_plot:
                        if self.multiple_lines and data_label == 'Differences':
                            cumulative_df = self.filter_results_df(
                                self.cumulative_differences_df, line)
                            if cumulative_df is not None:
                                x = cumulative_df[threshold_col].astype(str)
                                cumulative_y = cumulative_df[metric]

                                ax.plot(
                                    x,
                                    cumulative_y,
                                    marker='o',
                                    label=label
                                )
                                y_values.extend(cumulative_y.values)
                        else:
                            cumulative_df = self.filter_results_df(
                                self.cumulative_differences_df, line)
                            if cumulative_df is not None:
                                merged_df = pd.merge(
                                    filtered_df,
                                    cumulative_df[['Threshold',
                                                   metric] + self.id_columns],
                                    on=['Threshold'] + self.id_columns,
                                    suffixes=('', '_Cumulative')
                                )
                                x = merged_df[threshold_col].astype(str)
                                y = merged_df[metric]
                                cumulative_y = merged_df[f'{
                                    metric}_Cumulative']

                                colors = ['green' if val >=
                                          0 else 'red' for val in y]

                                bar_container = ax.bar(
                                    x, y, label='Individual Difference', color=colors)
                                y_values.extend(y.values)

                                line_plot, = ax.plot(
                                    x,
                                    cumulative_y,
                                    marker='o',
                                    color='blue',
                                    label='Cumulative Difference'
                                )
                                y_values.extend(cumulative_y.values)

                                table_data = pd.DataFrame({
                                    'Threshold': x,
                                    'Individual Diff': y,
                                    'Cumulative Diff': cumulative_y
                                })

                                table = ax.table(
                                    cellText=table_data[['Individual Diff', 'Cumulative Diff']].T.round(
                                        3).values,
                                    cellLoc='center',
                                    rowLoc='center',
                                    loc='bottom',
                                    bbox=[0.0, -0.25, 1.0, 0.125]
                                )

                                for (i, j), cell in table.get_celld().items():
                                    if i == 0:
                                        color = colors[j]
                                        cell.get_text().set_color(color)
                                    elif i == 1:
                                        cell.get_text().set_color('blue')
                                    cell.get_text().set_fontsize(8)

                            else:
                                logging.warning(
                                    f"No cumulative data found for line: {line}")
                                continue
                    else:
                        ax.plot(
                            filtered_df[threshold_col],
                            filtered_df[metric],
                            marker='o',
                            label=label
                        )
                        y_values.extend(filtered_df[metric].values)
                        if not (self.multiple_lines and data_label == 'Results'):
                            for x_val, y_val in zip(filtered_df[threshold_col], filtered_df[metric]):
                                ax.annotate(
                                    f"{y_val:.2f}",
                                    (x_val, y_val),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha='center',
                                    va='bottom',
                                    fontsize=8
                                )
            if y_values:
                min_y = min(y_values)
                max_y = max(y_values)
                y_range = max_y - min_y
                if y_range == 0:
                    y_range = 1
                margin = y_range * 0.1
                ax.set_ylim(min_y - margin, max_y + margin)

            if use_combined_plot:
                if self.multiple_lines and data_label == 'Differences':
                    ax.axhline(0, color='black', linewidth=0.8)
                    ax.yaxis.set_major_locator(MaxNLocator(10))
                    ax.grid(True, which='major', axis='y', linestyle='--')
                else:
                    ax.axhline(0, color='black', linewidth=0.8)
                    ax.yaxis.set_major_locator(MaxNLocator(10))
                    ax.grid(True, which='major', axis='y', linestyle='--')
                    ax.grid(True)
            else:
                ax.grid(True)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        number_prefix = f"{plot_number}_{sub_label}_"
        plot_filename = os.path.join(self.plots_directory, f"{self.dataset}_{
                                     number_prefix}{data_label}_Performance_Fairness_Metrics.png")
        plt.tight_layout()
        fig.suptitle(f"{self.dataset.capitalize()
                        } Performance Fairness {data_label}")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot saved: {plot_filename}")

    def plot_confusion_matrix_rates(self):

        if self.results_df is None or self.differences_df is None or self.cumulative_differences_df is None:
            logging.error(
                "Evaluation results or differences not loaded. Call load_evaluation_results() first.")
            raise ValueError("Evaluation results or differences not loaded.")

        group_names = ['Full', 'Priv', 'Prot']
        rate_types = ['Negative Rates', 'Positive Rates']
        negative_rates = ['True Negative Rate', 'False Positive Rate']
        positive_rates = ['True Positive Rate', 'False Negative Rate']
        rate_groups = [negative_rates, positive_rates]

        all_rates = []
        for group_prefix in group_names:
            for rate_list in rate_groups:
                for rate_name in rate_list:
                    metric_name = f"{group_prefix} {rate_name}"
                    all_rates.append(metric_name)

        global_max_diff = self._compute_global_max(
            self.differences_df, self.cumulative_differences_df, all_rates)

        plot_order = 1

        for group_prefix in group_names:
            for rate_type, rates in zip(rate_types, rate_groups):
                self._plot_confusion_rates(
                    self.results_df, group_prefix, rates, 'Threshold', rate_type, 'Results', 'A', plot_order)
                self._plot_confusion_rates(self.differences_df, group_prefix, rates, 'Current Threshold',
                                           rate_type, 'Differences', 'B', plot_order, use_combined_plot=True)
                plot_order += 1

    def _plot_confusion_rates(
        self,
        df,
        group_prefix,
        rates,
        threshold_col,
        rate_type,
        data_label,
        sub_label,
        plot_number,
        use_combined_plot=False
    ):

        fig, axes = plt.subplots(1, len(rates), figsize=(6 * len(rates), 5))
        plt.subplots_adjust(hspace=0.6, bottom=0.3)
        fig.suptitle(f"{self.dataset.capitalize()} {
                     group_prefix} {rate_type} {data_label}")

        if len(rates) == 1:
            axes = [axes]

        for idx, rate_name in enumerate(rates):
            ax = axes[idx]
            metric_name = f'{group_prefix} {rate_name}'
            ax.set_xlabel('Threshold')
            ax.set_ylabel(rate_name)
            y_values = []

            if threshold_col == 'Threshold' and not use_combined_plot:
                ax.set_xticks(self.threshold_values)

            for line in self.plots_list:
                filtered_df = self.filter_results_df(df, line)
                if filtered_df is not None and metric_name in filtered_df.columns:
                    label = f"{line['estimator_model']}_{
                        line['classification_model']}_{line['flip_method']}"
                    if use_combined_plot:
                        if self.multiple_lines and data_label == 'Differences':
                            cumulative_df = self.filter_results_df(
                                self.cumulative_differences_df, line)
                            if cumulative_df is not None:
                                x = cumulative_df[threshold_col].astype(str)
                                cumulative_y = cumulative_df[metric_name]

                                ax.plot(
                                    x,
                                    cumulative_y,
                                    marker='o',
                                    label=label
                                )
                                y_values.extend(cumulative_y.values)
                        else:
                            cumulative_df = self.filter_results_df(
                                self.cumulative_differences_df, line)
                            if cumulative_df is not None:
                                merged_df = pd.merge(
                                    filtered_df,
                                    cumulative_df[[
                                        'Threshold', metric_name] + self.id_columns],
                                    on=['Threshold'] + self.id_columns,
                                    suffixes=('', '_Cumulative')
                                )
                                x = merged_df[threshold_col].astype(str)
                                y = merged_df[metric_name]
                                cumulative_y = merged_df[f'{
                                    metric_name}_Cumulative']

                                colors = ['green' if val >=
                                          0 else 'red' for val in y]

                                bar_container = ax.bar(
                                    x, y, label='Individual Difference', color=colors)
                                y_values.extend(y.values)

                                line_plot, = ax.plot(
                                    x,
                                    cumulative_y,
                                    marker='o',
                                    color='blue',
                                    label='Cumulative Difference'
                                )
                                y_values.extend(cumulative_y.values)

                                table_data = pd.DataFrame({
                                    'Threshold': x,
                                    'Individual Diff': y,
                                    'Cumulative Diff': cumulative_y
                                })

                                table = ax.table(
                                    cellText=table_data[['Individual Diff', 'Cumulative Diff']].T.round(
                                        3).values,
                                    cellLoc='center',
                                    rowLoc='center',
                                    loc='bottom',
                                    bbox=[0.0, -0.25, 1.0, 0.125]
                                )

                                for (i, j), cell in table.get_celld().items():
                                    if i == 0:
                                        color = colors[j]
                                        cell.get_text().set_color(color)
                                    elif i == 1:
                                        cell.get_text().set_color('blue')
                                    cell.get_text().set_fontsize(8)
                    else:
                        ax.plot(
                            filtered_df[threshold_col],
                            filtered_df[metric_name],
                            marker='o',
                            label=label
                        )
                        y_values.extend(filtered_df[metric_name].values)
                        if not (self.multiple_lines and data_label == 'Results'):
                            for x_val, y_val in zip(filtered_df[threshold_col], filtered_df[metric_name]):
                                ax.annotate(
                                    f"{y_val:.2f}",
                                    (x_val, y_val),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha='center',
                                    va='bottom',
                                    fontsize=8
                                )
            if y_values:
                min_y = min(y_values)
                max_y = max(y_values)
                y_range = max_y - min_y
                if y_range == 0:
                    y_range = 1
                margin = y_range * 0.1
                ax.set_ylim(min_y - margin, max_y + margin)

            if use_combined_plot:
                if self.multiple_lines and data_label == 'Differences':
                    ax.axhline(0, color='black', linewidth=0.8)
                    ax.yaxis.set_major_locator(MaxNLocator(10))
                    ax.grid(True, which='major', axis='y', linestyle='--')
                else:
                    ax.axhline(0, color='black', linewidth=0.8)
                    ax.yaxis.set_major_locator(MaxNLocator(10))
                    ax.grid(True, which='major', axis='y', linestyle='--')
                    ax.grid(True)
            else:
                ax.grid(True)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        for idx in range(len(rates), len(axes)):
            fig.delaxes(axes[idx])

        number_prefix = f"{plot_number}_{sub_label}_"
        plot_filename = os.path.join(
            self.plots_directory,
            f"{self.dataset}_{number_prefix}{data_label}_Confusion_Matrix_{
                group_prefix}_{rate_type.replace(' ', '_')}.png"
        )
        plt.tight_layout()
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot saved: {plot_filename}")
