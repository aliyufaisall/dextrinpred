from sklearn.tree import DecisionTreeRegressor
from src.config import RANDOM_SEED


def train_dumb_baseline(X_train_engineered, y_train):
    """Trains a simple, shallow decision tree on our engineered features.

    Constraining max_depth ensures the baseline stays simple and serves as a
    clear performance floor.
    """
    # max_depth=3 keeps the tree small and easy to inspect
    model = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED)
    model.fit(X_train_engineered, y_train)
    return model
