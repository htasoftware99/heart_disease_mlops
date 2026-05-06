import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     AdaBoostClassifier)
from xgboost                 import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (accuracy_score, roc_auc_score,
                                     classification_report)
from src.logger              import get_logger
from src.custom_exception    import CustomException
from config.paths_config     import *
from utils.common_functions  import load_data, read_yaml

logger = get_logger(__name__)

MLFLOW_EXPERIMENT_NAME = "Heart_Disease_Prediction"

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF)": SVC(C=10, gamma=0.01, kernel="rbf", probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=5,
                                            min_samples_leaf=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                            min_samples_leaf=2, max_features="sqrt",
                                            random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                    max_depth=4, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    "XGBoost": XGBClassifier(objective="binary:logistic", n_estimators=200,
                             learning_rate=0.05, max_depth=5, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss",
                             random_state=42, verbosity=0),
}

MODEL_PARAMS = {
    "Logistic Regression":  {"max_iter": 1000, "random_state": 42},
    "K-Nearest Neighbours": {"n_neighbors": 7},
    "SVM (RBF)":            {"C": 10, "gamma": 0.01, "kernel": "rbf"},
    "Decision Tree":        {"criterion": "entropy", "max_depth": 5, "min_samples_leaf": 4},
    "Random Forest":        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2,
                             "max_features": "sqrt"},
    "Gradient Boosting":    {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 4},
    "AdaBoost":             {"n_estimators": 100, "learning_rate": 0.5},
    "XGBoost":              {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
                             "subsample": 0.8, "colsample_bytree": 0.8},
}


class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path, target_col="target"):
        self.train_path        = train_path
        self.test_path         = test_path
        self.model_output_path = model_output_path
        self.target_col        = target_col

        logger.info("ModelTraining initialized.")

    def load_and_split_data(self):
        try:
            logger.info(f"Train dataset loading  : {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Test dataset loading   : {self.test_path}")
            test_df  = load_data(self.test_path)

            X_train = train_df.drop(columns=[self.target_col])
            y_train = train_df[self.target_col]

            X_test  = test_df.drop(columns=[self.target_col])
            y_test  = test_df[self.target_col]

            logger.info(f"Dataset shape  -- Train: {X_train.shape} | Test: {X_test.shape}")
            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error occurred while loading data: {e}")
            raise CustomException("Data loading failed.", e)

    def _evaluate_model(self, name, model, X_train, X_test, y_train, y_test, cv=5):
        try:
            model.fit(X_train, y_train)

            y_pred  = model.predict(X_test)
            y_proba = (model.predict_proba(X_test)[:, 1]
                       if hasattr(model, "predict_proba") else None)

            acc_train = accuracy_score(y_train, model.predict(X_train))
            acc_test  = accuracy_score(y_test, y_pred)
            auc       = roc_auc_score(y_test, y_proba) if y_proba is not None else None

            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(cv, shuffle=True, random_state=42),
                scoring="accuracy",
            )

            logger.info("=" * 55)
            logger.info(f"  {name}")
            logger.info("=" * 55)
            logger.info(f"  Train Accuracy : {acc_train:.4f}")
            logger.info(f"  Test  Accuracy : {acc_test:.4f}")
            if auc is not None:
                logger.info(f"  ROC-AUC        : {auc:.4f}")
            logger.info(f"  CV ({cv}-fold) Mean : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            logger.info("\n" + classification_report(
                y_test, y_pred, target_names=["No Disease", "Has Disease"]
            ))

            return {
                "Model":     name,
                "Train Acc": acc_train,
                "Test Acc":  acc_test,
                "ROC-AUC":   auc,
                "CV Mean":   cv_scores.mean(),
                "CV Std":    cv_scores.std(),
                "_model":    model,
                "_y_pred":   y_pred,
                "_y_proba":  y_proba,
            }

        except Exception as e:
            logger.error(f"{name} error during evaluation: {e}")
            raise CustomException(f"{name} model evaluation failed.", e)

    def train_and_select(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("All models are being trained and evaluated...")
            results = []

            for name, model in MODELS.items():

                with mlflow.start_run(run_name=name, nested=True):

                    result = self._evaluate_model(name, model,
                                                  X_train, X_test,
                                                  y_train, y_test)
                    results.append(result)

                    mlflow.log_params(MODEL_PARAMS.get(name, {}))

                    mlflow.log_metric("train_accuracy", result["Train Acc"])
                    mlflow.log_metric("test_accuracy",  result["Test Acc"])
                    mlflow.log_metric("cv_mean",        result["CV Mean"])
                    mlflow.log_metric("cv_std",         result["CV Std"])
                    if result["ROC-AUC"] is not None:
                        mlflow.log_metric("roc_auc", result["ROC-AUC"])

                    mlflow.sklearn.log_model(
                        sk_model      = model,
                        artifact_path = name.replace(" ", "_").lower(),
                    )

            results_df = (
                pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                               for r in results])
                .sort_values("Test Acc", ascending=False)
                .reset_index(drop=True)
            )
            results_df.index += 1

            logger.info("\nModel Comparison Table:\n" + results_df.to_string())

            best_name  = results_df.iloc[0]["Model"]
            best_res   = next(r for r in results if r["Model"] == best_name)
            best_model = best_res["_model"]

            logger.info(f"\nBest Model    : {best_name}")
            logger.info(f"Test Accuracy : {best_res['Test Acc']:.4f}")
            logger.info(f"ROC-AUC       : {best_res['ROC-AUC']:.4f}")
            logger.info(f"CV Mean Acc.  : {best_res['CV Mean']:.4f} +/- {best_res['CV Std']:.4f}")

            return best_model, best_name, best_res, results_df

        except Exception as e:
            logger.error(f"Error occurred during model selection: {e}")
            raise CustomException("Model training and selection failed.", e)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved -> {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error occurred while saving model: {e}")
            raise CustomException("Model saving failed.", e)

    def _print_summary(self, best_name, best_res, n_models):
        logger.info("=" * 55)
        logger.info("     HEART DISEASE PREDICTION - SUMMARY")
        logger.info("=" * 55)
        logger.info(f"  Number of models evaluated   : {n_models}")
        logger.info(f"  Best model                   : {best_name}")
        logger.info(f"  Test Accuracy                : {best_res['Test Acc']:.4f}")
        logger.info(f"  ROC-AUC                      : {best_res['ROC-AUC']:.4f}")
        logger.info(f"  CV Mean Acc.                 : {best_res['CV Mean']:.4f} +/- {best_res['CV Std']:.4f}")
        logger.info(f"  Model saved                  : {MODEL_OUTPUT_PATH}")
        logger.info("=" * 55)

    def run(self):
        try:
            logger.info("=" * 55)
            logger.info("Model Training pipeline initializing...")

            X_train, y_train, X_test, y_test = self.load_and_split_data()

            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name="all_models_comparison"):

                best_model, best_name, best_res, results_df = self.train_and_select(
                    X_train, y_train, X_test, y_test
                )

                mlflow.log_param("best_model",        best_name)
                mlflow.log_metric("best_test_accuracy", best_res["Test Acc"])
                mlflow.log_metric("best_roc_auc",       best_res["ROC-AUC"])
                mlflow.log_metric("best_cv_mean",        best_res["CV Mean"])

                self.save_model(best_model)
                mlflow.log_artifact(self.model_output_path, artifact_path="best_model")

                mlflow.sklearn.log_model(
                    sk_model              = best_model,
                    artifact_path         = "best_model_registered",
                    registered_model_name = f"HeartDisease_{best_name.replace(' ', '_')}",
                )

                logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

            self._print_summary(best_name, best_res, n_models=len(MODELS))

            logger.info("Model Training pipeline successfully completed.")
            logger.info("=" * 55)

        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise CustomException("Model Training pipeline failed.", e)


if __name__ == "__main__":
    try:
        config     = read_yaml(CONFIG_PATH)
        target_col = config.get("data_processing", {}).get("target_column", "target")
    except Exception:
        target_col = "target"

    trainer = ModelTraining(
        train_path        = PROCESSED_TRAIN_DATA_PATH,
        test_path         = PROCESSED_TEST_DATA_PATH,
        model_output_path = MODEL_OUTPUT_PATH,
        target_col        = target_col,
    )
    trainer.run()