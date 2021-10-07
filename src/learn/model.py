from .interfaces import ModelInterface


class Model(ModelInterface):
    """
    Base class for each learning model.

    Get data as train, test and validation set.
    Learn from it and return the results.
    """

    def __init__(self):
        self.model = None

    def evaluate(self):
        pass

    def learn(self):
        pass

    def serve(self):
        pass

    def save_model(self):
        pass

    def load_model(self, filename):
        pass
