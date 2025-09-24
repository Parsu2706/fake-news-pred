import os
import pickle
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix

import mlflow
from mlflow.tracking import MlflowClient

import matplotlib.pyplot as plt
import seaborn as sns
import json


client = MlflowClient()
def main():
    try:

        data_path = "processed_data.csv"
        df = pd.read_csv(data_path)

        df = df.dropna(subset=['clean_text'])
        mlflow.set_experiment("Fake VS Real News Prediction")
        


        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            
        rf = RandomForestClassifier(random_state=42)

        param_distributions = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            }


        search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_distributions,
                n_iter=50,
                cv=3,
                n_jobs=-1,
                verbose=2,
                scoring='accuracy',
                random_state=42
            )
        

        with mlflow.start_run(): 

            mlflow.set_tag("Model" , "Random Forest Classifier")
            mlflow.sklearn.autolog(log_models=False)
                        
            #train
            search.fit(X_train, y_train)

            #Evaluation
            best_model = search.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            print("\nRandomizedSearch Completed!")
            print("Best Parameters:", search.best_params_)
            print("Best CV Accuracy:", round(search.best_score_ * 100, 2), "%")
            print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
            print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.2%}")
            print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.2%}")
            
            # Save metrics and logs best params 
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("Test Accuracy" , accuracy_score(y_test, y_test_pred))
            mlflow.log_metric("Train Accuracy" , accuracy_score(y_train  ,y_train_pred ))



            # save model locally and in mlflow
            model_path = "models/best_rf_model.pkl"
            vec_path = "models/tfidf_vectorizer.pkl"

            os.makedirs("models", exist_ok=True)
            with open("models/best_rf_model.pkl", "wb") as f:
                pickle.dump(search.best_estimator_, f)

            with open("models/tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(vectorizer, f)

            print("Model and vectorizer saved in 'models/'")

            mlflow.log_artifact(local_path = model_path , artifact_path='models')
            mlflow.log_artifact(local_path = vec_path , artifact_path="models")


            # saving confusion matrix in MLFLOW artifacts 
            cm = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            cm_path = "models/confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path, artifact_path='models')
            plt.close()
            
            # log the best_model

            mlflow.sklearn.log_model(best_model, artifact_path = "Best_rf_model")

        
          
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "train_accuracy": accuracy_score(y_train, y_train_pred)
            }
            with open("metrics.json", "w") as f:
                json.dump(metrics, f)

    except Exception as e:
        print(f"Error occurred: {e}")
        mlflow.end_run(status="FAILED") 


if __name__ == "__main__":
    main()
