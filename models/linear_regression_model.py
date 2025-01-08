import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

class LinearRegressionModel:
    def __init__(self):
        self.model_name = "Linear_Regression"

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
            if m % 100 == 0:
                print(f"Processed {m} samples...")

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
        print(f"Saving learning curve to {save_path}...")

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
        dataset_path = os.path.join('datasets', 'housing.csv')
        print("Reading dataset...")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        data = pd.read_csv(dataset_path)
        data = pd.get_dummies(data, columns=['ocean_proximity'])
        print("Dataset loaded and preprocessed")

        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)
        data = pd.DataFrame(data_imputed, columns=data.columns)
        print("Missing values handled")

        X = data.drop('median_house_value', axis=1)
        y = data['median_house_value']
        print("Splitting data into train and test sets")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Training the model...")
        pipeline.fit(X_train, y_train)
        print("Model trained successfully")

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        self.plot_learning_curves(pipeline, X_train, y_train, X_test, y_test)
        print("Learning curves generated")

        overfitting = mse_train < mse_test and (mse_test - mse_train) > mse_train * 0.2  # Arbitrary threshold for demo

        return {'mse_train': mse_train, 'mse_test': mse_test, 'overfitting': overfitting}

if __name__ == "__main__":
    lr_model = LinearRegressionModel()
    metrics = lr_model.train_and_evaluate_model()
    print(metrics)
