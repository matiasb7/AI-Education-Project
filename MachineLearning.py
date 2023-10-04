from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


class MachineLearning:
    def __init__(self, X_train, y_train):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train, y_train, test_size=0.25)

        # Models we are going to use with the params as dict to pass to the sklearn class
        # Structure for dict is 'model name' : [ModelClass from sklearn, { param=value, ....}]
        self.models = {
            'knn': [KNeighborsClassifier, {
                'n_neighbors': self.calculate_k()
            }],
            'random-forest': [RandomForestClassifier, {
                'random_state': 0
            }],
            'logistic-regression': [LogisticRegression, {
            }]
        }
        self.results = self.test_models()

    def calculate_k(self):
        """
        Calculate the k which returns the max score for the model
        """
        max_score = 0
        k_max_score = 0
        for k in range(1, 10):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            score = knn.score(self.X_test, self.y_test)
            if score > max_score:
                max_score = score
                k_max_score = k
        return k_max_score

    def calculate_results(self, y_pred, cv_results):
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        money_result = -0.5 * conf_matrix[0][1] - 0.25 * conf_matrix[1][0] + 250 * conf_matrix[1][1]
        return {
            'Score using Cross-validation': round(cv_results.mean(), 3),
            'Standard Deviation of score': round(cv_results.std() * 2, 3),
            'Money-Result': money_result,
        }

    def test_models(self):
        """
        Create model, fit and calculate scores.
        """
        results = {}
        for model, params in self.models.items():
            ml_model = params[0](**params[1])  # Initialize model class, example: KNeighborsClassifier()
            ml_model.fit(self.X_train, self.y_train)
            y_pred = ml_model.predict(self.X_test)
            cv_results = cross_val_score(ml_model, self.X_test, self.y_test, cv=5)
            results[model] = self.calculate_results(y_pred, cv_results)

        return results

    def predict_data(self, model, x_data):
        """
        Predict Y with the model selected and data.
        """
        selected_model = self.models[model]
        ml_model = selected_model[0]()
        ml_model.fit(self.X_train, self.y_train)
        return ml_model.predict(x_data)
