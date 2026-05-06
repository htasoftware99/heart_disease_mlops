import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path    = train_path
        self.test_path     = test_path
        self.processed_dir = processed_dir

        self.config    = read_yaml(config_path)
        self.dp_config = self.config["data_processing"]

        self.target_col = self.dp_config["target_column"]
        self.ohe_cols   = self.dp_config["ohe_columns"]   # ["cp", "slope", "thal"]

        os.makedirs(self.processed_dir, exist_ok=True)

        # State fitted on TRAIN, reused on TEST
        self.ohe_feature_names = []   # column names after get_dummies on train
        self.scaler = StandardScaler()

        logger.info("DataProcessor initialized.")

    # -------------------------------------------------------------------------
    # HELPER
    # -------------------------------------------------------------------------
    def _basic_clean(self, df):
        """Drop index artifact column and duplicate rows."""
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    # -------------------------------------------------------------------------
    # 1. PREPROCESS TRAIN  (fit + transform)
    # -------------------------------------------------------------------------
    def preprocess_train(self, df):
        """
        Mirrors notebook:
            df_model = pd.get_dummies(df, columns=["cp","slope","thal"], drop_first=True)
            X = df_model.drop("target", axis=1)
            X_train_sc = scaler.fit_transform(X_train)
        """
        try:
            logger.info("Preprocessing TRAIN data (Fit & Transform)...")

            df = self._basic_clean(df)

            # OHE: cp, slope, thal  (drop_first=True -- same as notebook)
            df = pd.get_dummies(df, columns=self.ohe_cols, drop_first=True)

            # Remember exact column layout so test can be aligned
            self.ohe_feature_names = [c for c in df.columns if c != self.target_col]
            logger.info(f"  Feature columns after OHE ({len(self.ohe_feature_names)}): "
                        f"{self.ohe_feature_names}")

            # StandardScaler: fit on train features
            X = df[self.ohe_feature_names].values
            X_scaled = self.scaler.fit_transform(X)
            df[self.ohe_feature_names] = X_scaled
            logger.info("  StandardScaler fitted and applied.")

            logger.info(f"Train preprocessing done. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Train preprocessing failed: {e}")
            raise CustomException("Train data preprocessing failed.", e)

    # -------------------------------------------------------------------------
    # 2. PREPROCESS TEST  (transform only -- no fitting)
    # -------------------------------------------------------------------------
    def preprocess_test(self, df):
        """
        Apply exactly the same transformations as train, using train's
        fitted OHE schema and scaler. No re-fitting.
        """
        try:
            logger.info("Preprocessing TEST data (Transform Only)...")

            df = self._basic_clean(df)

            # OHE with same settings
            df = pd.get_dummies(df, columns=self.ohe_cols, drop_first=True)

            # Align columns to train schema
            for col in self.ohe_feature_names:
                if col not in df.columns:
                    df[col] = 0                               # missing col -> 0
            extra = [c for c in df.columns
                     if c not in self.ohe_feature_names and c != self.target_col]
            df = df.drop(columns=extra, errors="ignore")      # unseen cats -> drop
            df = df[self.ohe_feature_names + [self.target_col]]  # enforce column order
            logger.info("  OHE schema aligned to train.")

            # StandardScaler: transform only
            X = df[self.ohe_feature_names].values
            df[self.ohe_feature_names] = self.scaler.transform(X)
            logger.info("  StandardScaler transform applied (train fit reused).")

            logger.info(f"Test preprocessing done. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Test preprocessing failed: {e}")
            raise CustomException("Test data preprocessing failed.", e)

    # -------------------------------------------------------------------------
    # 3. SAVE
    # -------------------------------------------------------------------------
    def save_data(self, df, file_path):
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Saved -> {file_path}  (shape: {df.shape})")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise CustomException("Error occurred while saving data.", e)

    # -------------------------------------------------------------------------
    # 4. MAIN PIPELINE
    # -------------------------------------------------------------------------
    def process(self):
        try:
            logger.info("=" * 55)
            logger.info("Data Processing pipeline starting...")

            train_df = load_data(self.train_path)
            test_df  = load_data(self.test_path)
            logger.info(f"Loaded -- Train: {train_df.shape} | Test: {test_df.shape}")

            # Fit on train, transform both
            train_df = self.preprocess_train(train_df)
            test_df  = self.preprocess_test(test_df)

            # Save processed splits
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,  PROCESSED_TEST_DATA_PATH)

            # Persist the fitted scaler so model_training.py can reuse it
            scaler_dir = os.path.join(self.processed_dir, "..", "models")
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved -> {scaler_path}")

            logger.info("Data Processing pipeline completed successfully.")
            logger.info("=" * 55)

        except Exception as e:
            logger.error(f"Data Processing pipeline error: {e}")
            raise CustomException("Data Processing pipeline error.", e)


if __name__ == "__main__":
    processor = DataProcessor(
        train_path    = TRAIN_FILE_PATH,
        test_path     = TEST_FILE_PATH,
        processed_dir = PROCESSED_DIR,
        config_path   = CONFIG_PATH,
    )
    processor.process()