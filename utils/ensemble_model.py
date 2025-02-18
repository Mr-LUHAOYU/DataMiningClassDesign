import lightgbm as lgb
import numpy as np
import pandas as pd
np.random.seed(42)


class DataSet:
    def __init__(self, X, y):
        self.X_true = X[y == 1]
        self.X_false = X[y == 0]
        self.y_true = y[y == 1]
        self.y_false = y[y == 0]

    def sample(self, true_num, false_num, replace=False, shuffled=True):
        indices_true = np.random.choice(
            len(self.y_true), true_num, replace=replace
        )
        indices_false = np.random.choice(
            len(self.y_false), false_num, replace=replace
        )
        X_sampled = pd.concat(
            [self.X_true.iloc[indices_true], self.X_false.iloc[indices_false]]
        )
        y_sampled = pd.concat(
            [self.y_true.iloc[indices_true], self.y_false.iloc[indices_false]]
        )
        if shuffled:
            indices_combined = np.random.permutation(len(X_sampled))
            X_sampled = X_sampled.iloc[indices_combined]
            y_sampled = y_sampled.iloc[indices_combined]
        return X_sampled, y_sampled

    def size(self):
        return len(self.y_true), len(self.y_false)


class ensemble_model:
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = [
            lgb.LGBMClassifier(random_state=42)
            for _ in range(n_models)
        ]

    def fit(self, X_train, y_train, X_valid, y_valid):
        data = DataSet(X_train, y_train)
        true_num, false_num = data.size()
        nums = int(true_num * 0.9)
        for model in self.models:
            X_train_, y_train_ = data.sample(nums, nums, replace=False)
            model.fit(
                X_train_, y_train_,
                eval_set=[(X_valid, y_valid)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(5)],
            )

    def predict(self, X_test):
        y_preds = []
        for model in self.models:
            y_pred = model.predict(
                X_test,
                num_iteration=model.best_iteration_
            )
            y_preds.append(y_pred)
        y_pred = sum(y_preds) / len(y_preds)
        return y_pred > 0.5
