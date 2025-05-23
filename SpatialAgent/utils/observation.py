class BaseObservation:
    """
    This is the base class for all observation classes.
    """

    def __init__(self, result_dict: dict) -> None:
        """
        Args:
            kwargs: the key-value pairs of the observation
        """
        for k, v in result_dict.items():
            setattr(self, k, v)
