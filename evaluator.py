# evaluator.py

import os
import logging
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score)

class Evaluator:

    def __init__(self, config):

        self.config = config
        self.results_directory = "results/"
        os.makedirs(self.results_directory, exist_ok=True)
        self.evaluation_results = []

        self.protected_encoded_value = 0
        self.privileged_encoded_value = 1
        self.negative_encoded_value = 0
        self.positive_encoded_value = 1

    def evaluate_model(self, model, X_test, y_test, sensitive_test, threshold):

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        metrics = self.compute_fairness_metrics(y_test, y_pred, sensitive_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_privileged = confusion_matrix(
            y_test[sensitive_test == self.privileged_encoded_value],
            y_pred[sensitive_test == self.privileged_encoded_value]
        )
        cm_protected = confusion_matrix(
            y_test[sensitive_test == self.protected_encoded_value],
            y_pred[sensitive_test == self.protected_encoded_value]
        )

        tn, fp, fn, tp = cm.ravel()
        full_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        full_fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        full_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        full_tpr = tp / (fn + tp) if (fn + tp) > 0 else 0.0

        tn_p, fp_p, fn_p, tp_p = cm_privileged.ravel()
        priv_tnr = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else 0.0
        priv_fpr = fp_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else 0.0
        priv_fnr = fn_p / (fn_p + tp_p) if (fn_p + tp_p) > 0 else 0.0
        priv_tpr = tp_p / (fn_p + tp_p) if (fn_p + tp_p) > 0 else 0.0

        tn_pr, fp_pr, fn_pr, tp_pr = cm_protected.ravel()
        prot_tnr = tn_pr / (tn_pr + fp_pr) if (tn_pr + fp_pr) > 0 else 0.0
        prot_fpr = fp_pr / (tn_pr + fp_pr) if (tn_pr + fp_pr) > 0 else 0.0
        prot_fnr = fn_pr / (fn_pr + tp_pr) if (fn_pr + tp_pr) > 0 else 0.0
        prot_tpr = tp_pr / (fn_pr + tp_pr) if (fn_pr + tp_pr) > 0 else 0.0

        evaluation_metrics = {
            'dataset': self.config['dataset']['name'],
            'processing_stage': self.config['dataset']['processing_stage'],
            'estimator_model': self.config['models']['estimator_model'],
            'classification_model': self.config['models']['classification_model'],
            'tune_hyperparameters': str(self.config['models']['tune_hyperparameters']),
            'num_trials': self.config['models']['hyperparameter_tuning']['num_trials'],
            'optimization_metric': self.config['models']['hyperparameter_tuning']['optimization_metric'],
            'flip_method': self.config['models']['flip_method'],
            'Threshold': threshold,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Matthews Correlation Coefficient': mcc,
            'ROC AUC Score': roc_auc,
            **metrics,
            'Full True Negative Rate': full_tnr,
            'Full False Positive Rate': full_fpr,
            'Full False Negative Rate': full_fnr,
            'Full True Positive Rate': full_tpr,
            'Priv True Negative Rate': priv_tnr,
            'Priv False Positive Rate': priv_fpr,
            'Priv False Negative Rate': priv_fnr,
            'Priv True Positive Rate': priv_tpr,
            'Prot True Negative Rate': prot_tnr,
            'Prot False Positive Rate': prot_fpr,
            'Prot False Negative Rate': prot_fnr,
            'Prot True Positive Rate': prot_tpr
        }

        self.evaluation_results.append(evaluation_metrics)
        logging.info(f"Evaluation completed for threshold {threshold}.")

    def compute_fairness_metrics(self, y_true, y_pred, sensitive_attr):

        protected_mask = sensitive_attr == self.protected_encoded_value
        privileged_mask = sensitive_attr == self.privileged_encoded_value

        p_protected = y_pred[protected_mask].mean()
        p_privileged = y_pred[privileged_mask].mean()
        statistical_parity = abs(p_protected - p_privileged)

        tpr_protected = self.true_positive_rate(y_true[protected_mask], y_pred[protected_mask])
        tpr_privileged = self.true_positive_rate(y_true[privileged_mask], y_pred[privileged_mask])
        equal_opportunity = abs(tpr_protected - tpr_privileged)

        fpr_protected = self.false_positive_rate(y_true[protected_mask], y_pred[protected_mask])
        fpr_privileged = self.false_positive_rate(y_true[privileged_mask], y_pred[privileged_mask])
        equalized_odds = 0.5 * (abs(tpr_protected - tpr_privileged) + abs(fpr_protected - fpr_privileged))

        fairness_metrics = {
            'Statistical Parity Difference': statistical_parity,
            'Equal Opportunity Difference': equal_opportunity,
            'Equalized Odds Difference': equalized_odds
        }

        return fairness_metrics

    def true_positive_rate(self, y_true, y_pred):

        tp = ((y_true == self.positive_encoded_value) & (y_pred == self.positive_encoded_value)).sum()
        fn = ((y_true == self.positive_encoded_value) & (y_pred == self.negative_encoded_value)).sum()
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def false_positive_rate(self, y_true, y_pred):

        fp = ((y_true == self.negative_encoded_value) & (y_pred == self.positive_encoded_value)).sum()
        tn = ((y_true == self.negative_encoded_value) & (y_pred == self.negative_encoded_value)).sum()
        if fp + tn == 0:
            return 0.0
        return fp / (fp + tn)

    def save_evaluation_results(self):

        results_df = pd.DataFrame(self.evaluation_results)
        results_filename = os.path.join(self.results_directory, 'evaluation_results.xlsx')

        match_columns = [
            'dataset', 'processing_stage', 'estimator_model', 'classification_model',
            'tune_hyperparameters', 'num_trials', 'optimization_metric', 'flip_method', 'Threshold'
        ]

        if os.path.isfile(results_filename):
            existing_sheets = pd.read_excel(results_filename, sheet_name=None)
        else:
            existing_sheets = {}

        if 'Results' in existing_sheets:
            existing_results = existing_sheets['Results']
            for _, new_row in results_df.iterrows():
                mask = (existing_results[match_columns] == new_row[match_columns]).all(axis=1)
                existing_results = existing_results[~mask]

            combined_results = pd.concat([existing_results, results_df], ignore_index=True)
        else:
            combined_results = results_df

        existing_sheets['Results'] = combined_results

        all_results = combined_results

        differences_list = []

        id_columns = [
            'dataset', 'processing_stage', 'estimator_model', 'classification_model',
            'tune_hyperparameters', 'num_trials', 'optimization_metric', 'flip_method'
        ]

        unique_ids = all_results[id_columns].drop_duplicates()

        for _, id_row in unique_ids.iterrows():

            id_mask = (all_results[id_columns] == id_row).all(axis=1)
            id_results = all_results[id_mask]

            id_results['Threshold'] = id_results['Threshold'].astype(float)
            id_results = id_results.sort_values('Threshold').reset_index(drop=True)

            for i in range(1, len(id_results)):
                prev_row = id_results.iloc[i - 1]
                curr_row = id_results.iloc[i]

                diff_metrics = id_row.copy()
                prev_threshold = prev_row['Threshold']
                curr_threshold = curr_row['Threshold']
                diff_metrics['Threshold'] = f"{curr_threshold:.2f} - {prev_threshold:.2f}"

                metric_columns = [
                    'Accuracy', 'F1 Score', 'Matthews Correlation Coefficient', 'ROC AUC Score',
                    'Statistical Parity Difference', 'Equal Opportunity Difference', 'Equalized Odds Difference',
                    'Full True Negative Rate', 'Full False Positive Rate', 'Full False Negative Rate', 'Full True Positive Rate',
                    'Priv True Negative Rate', 'Priv False Positive Rate', 'Priv False Negative Rate', 'Priv True Positive Rate',
                    'Prot True Negative Rate', 'Prot False Positive Rate', 'Prot False Negative Rate', 'Prot True Positive Rate'
                ]

                for metric in metric_columns:
                    diff_metrics[metric] = curr_row[metric] - prev_row[metric]

                differences_list.append(diff_metrics)

        if differences_list:
            differences_df = pd.DataFrame(differences_list)
        else:
            differences_df = pd.DataFrame()

        if 'Differences' in existing_sheets:
            existing_differences = existing_sheets['Differences']
            for _, diff_row in differences_df.iterrows():
                mask = (
                    (existing_differences[id_columns] == diff_row[id_columns]).all(axis=1) &
                    (existing_differences['Threshold'] == diff_row['Threshold'])
                )
                existing_differences = existing_differences[~mask]

            combined_differences = pd.concat([existing_differences, differences_df], ignore_index=True)
        else:
            combined_differences = differences_df

        existing_sheets['Differences'] = combined_differences

        if not combined_differences.empty:
            cumulative_differences_df = combined_differences.copy()
            id_columns_with_threshold = id_columns + ['Threshold']
            cumulative_differences_df.sort_values(by=id_columns_with_threshold, inplace=True)

            cumulative_differences_df['Current Threshold'] = cumulative_differences_df['Threshold'].apply(
                lambda x: float(x.split(' - ')[0])
            )

            metric_columns = [
                'Accuracy', 'F1 Score', 'Matthews Correlation Coefficient', 'ROC AUC Score',
                'Statistical Parity Difference', 'Equal Opportunity Difference', 'Equalized Odds Difference',
                'Full True Negative Rate', 'Full False Positive Rate', 'Full False Negative Rate', 'Full True Positive Rate',
                'Priv True Negative Rate', 'Priv False Positive Rate', 'Priv False Negative Rate', 'Priv True Positive Rate',
                'Prot True Negative Rate', 'Prot False Positive Rate', 'Prot False Negative Rate', 'Prot True Positive Rate'
            ]

            cumulative_differences_df[metric_columns] = cumulative_differences_df.groupby(id_columns)[metric_columns].cumsum()

            if 'Cumulative Differences' in existing_sheets:
                existing_cumulative = existing_sheets['Cumulative Differences']
                for _, cum_row in cumulative_differences_df.iterrows():
                    mask = (
                        (existing_cumulative[id_columns] == cum_row[id_columns]).all(axis=1) &
                        (existing_cumulative['Threshold'] == cum_row['Threshold'])
                    )
                    existing_cumulative = existing_cumulative[~mask]

                combined_cumulative = pd.concat([existing_cumulative, cumulative_differences_df], ignore_index=True)
            else:
                combined_cumulative = cumulative_differences_df

            existing_sheets['Cumulative Differences'] = combined_cumulative
        else:
            existing_sheets['Cumulative Differences'] = pd.DataFrame()

        with pd.ExcelWriter(results_filename, engine='openpyxl', mode='w') as writer:
            for sheet_name, df in existing_sheets.items():
                df.to_excel(writer, index=False, sheet_name=sheet_name)

        logging.info(f"Evaluation results saved to {results_filename}")