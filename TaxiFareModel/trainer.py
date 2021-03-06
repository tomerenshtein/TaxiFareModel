# imports
from TaxiFareModel.data import get_data, clean_data, holdout
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import cross_validate



class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[FR][PAR][tomerenshtein] TaxiFare 01"


    def __init__(self, X, y, model):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.model = model
        self.cv_results = None

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])],
            remainder="drop")

        self.pipeline = Pipeline([('preproc', preproc_pipe),
                                  ('model', self.model)])
        return self

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()


    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)


    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)


    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.cv_results = cross_validate(self.pipeline, self.X, self.y, cv=3)
        self.mlflow_log_metric("cross_val_score",
                               self.cv_results['test_score'].mean())
        print(self.cv_results['test_score'].mean())

        self.pipeline.fit(self.X, self.y)

        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        self.mlflow_log_param("model", self.pipeline[1])

        print(rmse)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, f'{self.pipeline[1]}.joblib')




if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df_clean = clean_data(df)

    # set X and y
    y = df.pop("fare_amount")
    X = df

    # hold out
    X_train, X_test, y_train, y_test = holdout(X,y)


    for model_type in [LinearRegression(), SVR()]:



        # train
        model = Trainer(X,y,model_type)

        #run
        model.run()

        # evaluate
        model.evaluate(X_test, y_test)

        #save model
        model.save_model()

experiment_id = model.mlflow_experiment_id
print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
