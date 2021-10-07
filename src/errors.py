class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class ModelError(Error):
    """Exception raised for errors in the input.

    Attributes:
        id -- error unique identifier
        message -- explanation of the error
    """

    def __init__(self, identifier, message=None):
        self._id = identifier
        self._message = message

    @property
    def message(self):
        return self._message


class ModelErrors:
    empty_dataframe = ModelError(1, "Empty dataframe")
