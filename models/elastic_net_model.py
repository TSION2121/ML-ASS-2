import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

class ElasticNetModel:
    def __init__(self):
        self.model_name = "Elastic_Net"

    def download_dataset(self):
        print("Downloading dataset...")
        if not os.path.exists('datasets/housing.csv'):
            os.system('kaggle datasets download -d camnugent/california-housing-prices -p datasets --unzip')
            print("Dataset downloaded")

    def plot_learning_curves(self, model, X_train, y_train, X_test, y_test):
        train_errors, test_errors = [], []
        print("Generating learning curves...")
        for m in range(1, len(X_train)):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_test_predict = model.predict(X_test)
            train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
            test_errors.append(mean_squared_error(y_test, y_test_predict))

        plt.figure(figsize=(10, 6))
        plt.plot(train_errors, "r-+", linewidth=2, label="Train")
        plt.plot(test_errors, "b-", linewidth=3, label="Test")
        plt.xlabel("Training Set Size")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.title(f"Learning Curves for {self.model_name}")

        static_dir = os.path.join(os.getcwd(), 'static')
        os.makedirs(static_dir, exist_ok=True)
        save_path = os.path.join(static_dir, f'learning_curve_{self.model_name}.png')

        try:
            plt.savefig(save_path)
            print(f"Learning curve saved successfully at {save_path}")
        except Exception as e:
            print(f"Error saving learning curve: {e}")
        finally:
            plt.close()
            print("Plot closed.")

    def train_and_evaluate_model(self):
        self.download_dataset()
        dataset_path = 'datasets/housing.csv'
        print("Reading dataset...")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        data = pd.read_csv(dataset_path)
        print("Dataset loaded")  # Debugging print

        data = pd.get_dummies(data, columns=['ocean_proximity'])
        print("One-hot encoding completed")  # Debugging print

        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)
        data = pd.DataFrame(data_imputed, columns=data.columns)
        print("Missing values handled")  # Debugging print

        data_sampled = data.sample(frac=0.1, random_state=42)
        print("Dataset sampled")  # Debugging print

        X = data_sampled.drop('median_house_value', axis=1)
        y = data_sampled['median_house_value']
        print("Data preprocessing completed")  # Debugging print

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', ElasticNet(max_iter=20000))
        ])
        print("Pipeline created")  # Debugging print

        param_grid = {
            'regressor__alpha': [0.1, 1],
            'regressor__l1_ratio': [0.5, 0.9]
        }

        print("Starting GridSearchCV...")  # Debugging print
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        print("Grid search completed")  # Debugging print

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        best_pipeline = grid_search.best_estimator_
        print("Training the model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_pipeline.fit(X_train, y_train)
        print("Model trained successfully")

        y_pred_en = best_pipeline.predict(X_test)
        mse_en = mean_squared_error(y_test, y_pred_en)
        print(f"MSE Elastic Net: {mse_en}")  # Debugging print

        self.plot_learning_curves(best_pipeline, X_train, y_train, X_test, y_test)
        print("Learning curves generated")

        return {'mse_en': mse_en, 'best_params': best_params}

if __name__ == "__main__":
    en_model = ElasticNetModel()
    metrics = en_model.train_and_evaluate_model()
    print(metrics)
