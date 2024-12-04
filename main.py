# main.py

import argparse
import json
import logging
import sys
import os
import copy

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import Evaluator
from visualizer import Visualizer

script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description="Fairness in Machine Learning Models")
parser.add_argument('--generate_plots', action='store_true',
                    help='Generate plots from existing results')
args = parser.parse_args()

with open(os.path.join(script_dir, "config_log.json"), 'r') as config_file:
    config_log = json.load(config_file)

with open(os.path.join(script_dir, "config_general.json"), 'r') as config_file:
    config_general = json.load(config_file)

with open(os.path.join(script_dir, "config_datasets.json"), 'r') as config_file:
    config_datasets = json.load(config_file)

with open(os.path.join(script_dir, "config_models_flips.json"), 'r') as config_file:
    config_models_flips = json.load(config_file)

configs = []

for data_dict in config_datasets:
    for model_dict in config_models_flips:
        conf = copy.deepcopy(config_general)
        conf['dataset'].update(data_dict)
        conf['models'].update(model_dict)
        configs.append(conf)


def setup_logging(log_level, log_file):

    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main(
    configs=configs,
    generate_plots=args.generate_plots
):

    setup_logging(config_log['logging']['log_level'],
                  config_log['logging']['log_file'])
    logging.info("Program started")

    if generate_plots:
        with open(os.path.join(script_dir, "config_visual.json"), 'r') as config_file:
            config_visual = json.load(config_file)

        visualizer = Visualizer(config_visual)
        visualizer.load_evaluation_results()
        visualizer.plot_confusion_matrix_rates()
        visualizer.plot_metrics()
        logging.info("Plots generated successfully")

    else:

        for config in configs:

            data_processor = DataProcessor(config)
            data_processor.load_original_data()
            data_processor.encode_data()
            data_processor.scale_data()
            data_processor.split_data()

            model_trainer = ModelTrainer(config)
            evaluator = Evaluator(config)

            data_processor.set_unfairness_parameters()
            if config['models']['flip_method'] in ['Min', 'Max']:
                data_processor.train_estimator()
            for threshold in config['models']['thresholds']:
                unfair_train_data = data_processor.introduce_unfairness(
                    threshold,
                    method=config['models']['flip_method']
                )

                classification_model = model_trainer.train_model(
                    train_data=unfair_train_data,
                    model_name=config['models']['classification_model'],
                    model_identifier='classifier',
                    tune_hyperparameters=config['models']['tune_hyperparameters'],
                    optimization_metric=config['models']['hyperparameter_tuning']['optimization_metric'],
                    flip_method=f'_{config['models']['flip_method']}',
                    threshold=f'_{str(threshold)}'
                )

                evaluator.evaluate_model(
                    model=classification_model,
                    X_test=data_processor.X_test,
                    y_test=data_processor.y_test,
                    sensitive_test=data_processor.sensitive_test,
                    threshold=threshold
                )
            evaluator.save_evaluation_results()


if __name__ == '__main__':

    main(
        generate_plots=True
    )
