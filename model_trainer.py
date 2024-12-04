# model_trainer.py

import os
import json
import logging
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class ModelTrainer:

    def __init__(self, config):

        self.config = config
        models_directory = "models"
        os.makedirs(models_directory, exist_ok=True)
        dataset_dir = self.config["dataset"]["name"]
        estim_dir = self.config["models"]["estimator_model"]
        self.model_data_estim_dir = os.path.join(
            models_directory,
            dataset_dir,
            estim_dir
        )
        os.makedirs(self.model_data_estim_dir, exist_ok=True)

        hyperparams_path = os.path.join("Codigos", self.config.get(
            'models', {}).get('hyperparameters_path', 'config_hyperparameters.json'))
        if not os.path.isfile(hyperparams_path):
            logging.error(f"Hyperparameters file not found at {
                          hyperparams_path}")
            raise FileNotFoundError(
                f"Hyperparameters file not found at {hyperparams_path}")

        with open(hyperparams_path, 'r') as f:
            self.hyperparameters = json.load(f)

    def train_model(self,
                    train_data,
                    model_name,
                    model_identifier,
                    tune_hyperparameters,
                    optimization_metric,
                    flip_method='',
                    threshold=''
                    ):

        X_train, y_train = train_data

        model_filename = os.path.join(
            self.model_data_estim_dir,
            f"{model_name}_{model_identifier}{flip_method}{threshold}.joblib"
        )
        if os.path.isfile(model_filename):
            model = joblib.load(model_filename)
            logging.info(f"Loaded existing {
                         model_identifier} model from {model_filename}")
            return model

        if tune_hyperparameters:
            model = self.tune_hyperparameters(
                X_train, y_train, model_name, optimization_metric)
        else:
            model = self.initialize_model(model_name)
            model.fit(X_train, y_train)
            logging.info(
                f"Model {model_name} trained without hyperparameter tuning.")

        joblib.dump(model, model_filename)
        logging.info(f"Model saved to {model_filename}")

        return model

    def initialize_model(self, model_name):

        if model_name == "RandomForest":
            return RandomForestClassifier()
        elif model_name == "MLPClassifier":
            return MLPClassifier(max_iter=500)
        elif model_name == "DecisionTree":
            return DecisionTreeClassifier()
        elif model_name == "LogisticRegression":
            return LogisticRegression(max_iter=1000, solver='saga')
        elif model_name == "SVC":
            return SVC(probability=True)
        elif model_name == "GaussianNB":
            return GaussianNB()
        else:
            logging.error(f"Model '{model_name}' is not supported.")
            raise ValueError(f"Model '{model_name}' is not supported.")

    def tune_hyperparameters(self, X_train, y_train, model_name, optimization_metric):

        num_trials = self.config['models']['hyperparameter_tuning']['num_trials']
        scorer = self.get_scorer(optimization_metric)

        if model_name in ["RandomForest", "DecisionTree", "LogisticRegression"]:
            param_distributions = self.get_param_distributions(model_name)
            model = self.initialize_model(model_name)
            search = RandomizedSearchCV(
                model,
                param_distributions=param_distributions,
                n_iter=num_trials,
                scoring=scorer,
                cv=5,
                n_jobs=-1,
                # random_state=42
            )
            search.fit(X_train, y_train)
            logging.info(f"Best hyperparameters for {
                         model_name}: {search.best_params_}")
            best_model = search.best_estimator_
            return best_model
        elif model_name in ["MLPClassifier", "SVC"]:
            def objective(trial):
                model = self.initialize_model_with_trial(trial, model_name)
                score = cross_val_score(
                    model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1)
                return score.mean()

            sampler = TPESampler(
                # seed=42
            )
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=num_trials)

            logging.info(f"Best hyperparameters for {
                         model_name}: {study.best_params}")

            best_model = self.initialize_model_with_params(
                model_name, study.best_params)
            best_model.fit(X_train, y_train)

            return best_model
        elif model_name == "GaussianNB":
            logging.info(f"No hyperparameters to tune for {
                         model_name}. Training default model.")
            model = self.initialize_model(model_name)
            model.fit(X_train, y_train)
            return model
        else:
            logging.error(
                f"Unsupported model for hyperparameter tuning: {model_name}")
            raise ValueError(
                f"Unsupported model for hyperparameter tuning: {model_name}")

    def get_param_distributions(self, model_name):

        if model_name in self.hyperparameters:
            param_dist = self.hyperparameters[model_name]
            if isinstance(param_dist, list):
                return param_dist
            else:
                return param_dist
        else:
            logging.error(f"Hyperparameters for model '{
                          model_name}' are not defined in config_hyperparameters.json.")
            raise ValueError(f"Hyperparameters for model '{
                             model_name}' are not defined.")

    def initialize_model_with_trial(self, trial, model_name):

        if model_name in self.hyperparameters:
            params_config = self.hyperparameters[model_name]
            params = {}
            for param_name, param_options in params_config.items():
                param_type = param_options['type']
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_options['low'],
                        param_options['high'],
                        step=param_options.get('step', 1)
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_options['low'],
                        param_options['high'],
                        step=param_options.get('step', None),
                        log=param_options.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_options['choices']
                    )
                else:
                    logging.error(f"Unsupported parameter type '{
                                  param_type}' for {param_name}.")
                    raise ValueError(f"Unsupported parameter type '{
                                     param_type}' for {param_name}.")
            return self.initialize_model_with_params(model_name, params)
        else:
            logging.error(f"Hyperparameters for model '{
                          model_name}' are not defined in config_hyperparameters.json.")
            raise ValueError(f"Hyperparameters for model '{
                             model_name}' are not defined.")

    def initialize_model_with_params(self, model_name, params):

        if model_name == "MLPClassifier":
            params.setdefault('max_iter', 500)
            return MLPClassifier(**params)
        elif model_name == "SVC":
            params.setdefault('probability', True)
            return SVC(**params)
        elif model_name == "LogisticRegression":
            params.setdefault('solver', 'saga')
            params.setdefault('max_iter', 1000)
            return LogisticRegression(**params)
        elif model_name == "RandomForest":
            return RandomForestClassifier(**params)
        elif model_name == "DecisionTree":
            return DecisionTreeClassifier(**params)
        else:
            logging.error(
                f"Model '{model_name}' is not supported for initialization with parameters.")
            raise ValueError(
                f"Model '{model_name}' is not supported for initialization with parameters.")

    def get_scorer(self, optimization_metric):

        if optimization_metric == 'accuracy':
            return make_scorer(accuracy_score)
        elif optimization_metric == 'f1_score':
            return make_scorer(f1_score)
        elif optimization_metric == 'matthews_corrcoef':
            return make_scorer(matthews_corrcoef)
        else:
            logging.error(f"Unsupported optimization metric: {
                          optimization_metric}")
            raise ValueError(f"Unsupported optimization metric: {
                             optimization_metric}")
