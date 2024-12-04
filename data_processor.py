# data_processor.py

import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from model_trainer import ModelTrainer

class DataProcessor:

    def __init__(self, config):


        self.config = config
        self.original_dataset = None
        self.X_train = None
        self.y_train = None

        self.protected_encoded_value = 0
        self.privileged_encoded_value = 1
        self.negative_encoded_value = 0
        self.positive_encoded_value = 1

        self.dataset_path = os.path.join(
            'datasets',
            self.config['dataset']['name'],
            f'{self.config['dataset']['name']}.{self.config['dataset']['extension']}'
            )

    def load_original_data(self):

        try:
            self.original_dataset = pd.read_csv(self.dataset_path)
            logging.info(f"Dataset loaded successfully from {self.dataset_path}")
        except FileNotFoundError:
            logging.error(f"Dataset file not found at {self.dataset_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

        if self.config['dataset']['name'] == "bank":
            self.original_dataset['day'] = self.original_dataset['day'].astype(str)
            self.original_dataset['previous_contact'] = self.original_dataset['pdays'].apply(lambda x: 0 if x == -1 else 1)
            self.original_dataset['pdays'] = self.original_dataset['pdays'].apply(lambda x: 0 if x == -1 else x)
        if self.config['dataset']['name'] == 'adult':
            self.original_dataset.drop(columns=['educational-num'],inplace=True)

    def encode_data(self):

        if os.path.exists(os.path.join(
            os.path.dirname(self.dataset_path),
            f'encoded_{os.path.basename(self.dataset_path)}'
            )):
            logging.info("This dataset is already encoded")

        else:

            if self.original_dataset is None:
                logging.error("Dataset not loaded. Call load_data() first.")
                raise ValueError("Dataset not loaded.")

            df = self.original_dataset.copy()
            config = self.config['dataset']

            label_name = config['label_name']
            negative_label = config['negative_label']
            positive_label = (set(df[label_name].unique()) - {negative_label}).pop()

            sensitive_attr = config['sensitive_attribute']
            protected_value = config['protected_value']
            privileged_value = (set(df[sensitive_attr].unique()) - {protected_value}).pop()

            self.label_mapping = {
                negative_label: self.negative_encoded_value,
                positive_label: self.positive_encoded_value
            }

            self.sensitive_mapping = {
                protected_value: self.protected_encoded_value,
                privileged_value: self.privileged_encoded_value
            }

            if self.config['dataset']['name'] == "bank":
                self.sensitive_mapping['divorced'] = 0

            df[label_name] = df[label_name].map(self.label_mapping)
            df[sensitive_attr] = df[sensitive_attr].map(self.sensitive_mapping)

            if self.config['dataset']['name'] == "compas":
                df.to_csv(os.path.join(
                    os.path.dirname(self.dataset_path),
                    f'encoded_{os.path.basename(self.dataset_path)}'
                    ),index=False)
                return

            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            le = LabelEncoder()
            for col in categorical_cols:
                if df[col].nunique() == 2:
                    df[col] = le.fit_transform(df[col])

            one_hot_encoder = OneHotEncoder(sparse_output=False,drop='first')
            one_hot_categorical_cols = [col for col in categorical_cols if df[col].nunique() > 2]
            encoded_data = one_hot_encoder.fit_transform(df[one_hot_categorical_cols])
            encoded_col_names = one_hot_encoder.get_feature_names_out(one_hot_categorical_cols)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_col_names, index=df.index)
            df = df.drop(columns=one_hot_categorical_cols).join(encoded_df)

            df.to_csv(os.path.join(
                os.path.dirname(self.dataset_path),
                f'encoded_{os.path.basename(self.dataset_path)}'
                ),index=False)

            logging.info("Data encoding completed.")

    def scale_data(self):

        if self.config['dataset']['name'] == "compas": return

        if os.path.exists(os.path.join(
            os.path.dirname(self.dataset_path),
            f'scaled_{os.path.basename(self.dataset_path)}'
            )):
            logging.info("This dataset is already encoded and scaled")

        else:

            df = pd.read_csv(os.path.join(
                os.path.dirname(self.dataset_path),
                f'encoded_{os.path.basename(self.dataset_path)}'
                ))

            if df is None:
                logging.error("Data not encoded. Call encode_data() first.")
                raise ValueError("Data not encoded.")

            numerical_cols = self.original_dataset.select_dtypes(include=[np.number]).columns.tolist()

            if self.config['dataset']['name'] == "bank":
                numerical_cols.remove('previous_contact')

            scaler = MinMaxScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

            df.to_csv(os.path.join(
                os.path.dirname(self.dataset_path),
                f'scaled_{os.path.basename(self.dataset_path)}'
                ),index=False)

            logging.info("Data scaling completed, excluding one-hot encoded columns.")

    def split_data(self):

        config = self.config['dataset']

        df = pd.read_csv(os.path.join(
            os.path.dirname(self.dataset_path),
            f"{config['processing_stage']}_{os.path.basename(self.dataset_path)}"
            ))

        if df is None:
            logging.error("Data not scaled. Call scale_data() first.")
            raise ValueError("Data not scaled.")

        label_name = config['label_name']
        sensitive_attr = config['sensitive_attribute']
        test_size = config['test_size']
        random_state = config['random_state']

        X = df.drop(columns=[label_name])
        y = df[label_name]
        sensitive = X[sensitive_attr]

        stratify_col = pd.Series(list(zip(y, sensitive)))

        self.X_train, self.X_test, self.y_train, self.y_test, self.sensitive_train, self.sensitive_test = train_test_split(
            X, y, sensitive, test_size=test_size, random_state=random_state, stratify=stratify_col
        )

        logging.info("Data splitting completed.")

    def set_unfairness_parameters(self):

        if self.X_train is None or self.y_train is None:
            logging.error("Data not split. Call split_data() first.")
            raise ValueError("Data not split.")

        train_df = self.X_train.copy()
        train_df[self.config['dataset']['label_name']] = self.y_train.copy()

        self.condition = (
            ((train_df[self.config['dataset']['sensitive_attribute']] == self.protected_encoded_value) &
            (train_df[self.config['dataset']['label_name']] == self.positive_encoded_value)) |
            ((train_df[self.config['dataset']['sensitive_attribute']] == self.privileged_encoded_value) &
            (train_df[self.config['dataset']['label_name']] == self.negative_encoded_value))
        )

        self.indices_to_flip = train_df[self.condition].index

    def train_estimator(self):

        model_trainer = ModelTrainer(self.config)
        train_data = (self.X_train, self.y_train)

        estimator_model = model_trainer.train_model(
            train_data=train_data,
            model_name=self.config['models']['estimator_model'],
            model_identifier='estimator',
            tune_hyperparameters=self.config['models']['tune_hyperparameters'],
            optimization_metric=self.config['models']['hyperparameter_tuning']['optimization_metric']
        )

        probabilities = estimator_model.predict_proba(self.X_train)[:, 1]
        condition_probabilities = probabilities[self.condition]
        confidence_scores = np.abs(condition_probabilities - 0.5)
        self.sorted_indices = self.indices_to_flip[np.argsort(confidence_scores)]

    def introduce_unfairness(self, threshold, method):

        unfair_train_data = self.y_train.copy()

        num_to_flip = int(len(self.indices_to_flip) * threshold)

        if method == 'Random':
            indices_flipped = np.random.choice(self.indices_to_flip, size=num_to_flip, replace=False)
        elif method == 'Max':
            num_to_flip = int(threshold * len(self.indices_to_flip))
            if num_to_flip > 0:
                indices_flipped = self.sorted_indices[-num_to_flip:]
            else:
                indices_flipped = pd.Index([], dtype=self.sorted_indices.dtype)
            self.indices_flipped = indices_flipped
        elif method == 'Min':
            indices_flipped = self.sorted_indices[:num_to_flip]
        else:
            logging.error(f"Unknown flip method: {method}")
            raise ValueError(f"Unknown flip method: {method}")

        unfair_train_data.loc[indices_flipped] = 1 - unfair_train_data.loc[indices_flipped]

        self.y_unfair_train = unfair_train_data
        unfair_train_data = (self.X_train, self.y_unfair_train)

        logging.info(f"Unfairness introduced with threshold {threshold} using method {method}.")

        return unfair_train_data