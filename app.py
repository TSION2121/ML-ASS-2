from flask import Flask, render_template
from models.linear_regression_model import LinearRegressionModel
from models.elastic_net_model import ElasticNetModel

app = Flask(__name__)

@app.route('/')
def index():
    lr_model = LinearRegressionModel()
    lr_metrics = lr_model.train_and_evaluate_model()

    en_model = ElasticNetModel()
    en_metrics = en_model.train_and_evaluate_model()

    return render_template('index.html', lr_metrics=lr_metrics, en_metrics=en_metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
