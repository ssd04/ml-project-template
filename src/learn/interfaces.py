import abc


class ModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "learn")
            and callable(subclass.learn)
            and hasattr(subclass, "serve")
            and callable(subclass.evaluate)
            and hasattr(subclass, "save_model")
            and callable(subclass.save_model)
            and hasattr(subclass, "load_model")
            and callable(subclass.load_model)
            or NotImplemented
        )

    @abc.abstractmethod
    def learn(self):
        """Main training part."""
        raise NotImplementedError

    @abc.abstractmethod
    def serve(self, X):
        """Serve predictions based on input data X."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, filename: str):
        """Save the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, filename: str):
        """Load an already trained module."""
        raise NotImplementedError
