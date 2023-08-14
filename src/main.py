from src.preprocessor import Preprocessor
from src.model import ChurnClassifier
import pandas as pd
import logging
from xgboost import XGBClassifier

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

INPUT_PATH = "../data/transactions_dataset.csv"
RELATIONSHIP_PATH = "../data/sales_client_relationship_dataset.csv"
OUTPUT_PATH = "../data/results.csv"


if __name__ == "__main__":
    # Reading csv file
    preprocessor = Preprocessor(INPUT_PATH, nrows=1000)

    # Preprocessing df
    df = preprocessor.full_preprocessing()
    relationship = pd.read_csv(RELATIONSHIP_PATH)
    relationship = pd.get_dummies(relationship[["client_id", "quali_relation"]])
    df = df.merge(relationship, how="left", on="client_id")

    classifier = ChurnClassifier(df)

    # Train the XGBoost model
    classifier.train(XGBClassifier(), smote=True)
    # threshold
    threshold = classifier.select_threshold()
    # Evaluate the performance of the model on the testing set
    classifier.evaluate()
    classifier.auc()
    classifier.plot_feature_importance()
    classifier.shapley_values()
    classifier.output(OUTPUT_PATH)
