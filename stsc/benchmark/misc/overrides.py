def overrides(base_class):
    """
    Function override annotation taken from https://stackoverflow.com/a/54526724.
    Corollary to @abc.abstractmethod where the override is not of an abstractmethod.

    Checks if annotated function overrides an actually existing function in given base class.
    Also for documentation purposes.
    """
    def confirm_override(func):
        if func.__name__ not in dir(base_class):
            raise NotImplementedError(f"Function {func.__name__} tries to override undefined function in base class {base_class}")

        def f():
            pass

        attr = getattr(base_class, func.__name__)
        if type(attr) is not type(f):
            raise NotImplementedError(f"Type Mismatch: Function {func.__name__} tries to overrides symbol of type {type(attr)} in base class {base_class}.")
        return func
    return confirm_override