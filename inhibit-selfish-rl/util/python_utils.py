

def forcefunction(func):
    """Decorator to force a function to be implemented in a subclass."""
    func.__forcefunction__ = True
    return func


# def forceproperty(func):
#     """Decorator to force a property to be implemented in a subclass."""
#     func.__forceproperty__ = True
#     return property(func)


class ForceClassFormatError(Exception):
    pass


class ForceFunctionMeta(type):
    """Metaclass to check that:
    - all functions decorated with @forcefunction are implemented in subclasses
    - all properties decorated with @forceproperty are implemented in subclasses
    """
    def __new__(mcs, name, bases, namespace):
        for base in bases:
            for attr_name, attr_value in base.__dict__.items():
                if hasattr(attr_value, '__forcefunction__'):
                    if attr_name not in namespace:
                        raise ForceClassFormatError(f"Class {name} must implement function {attr_name} "
                                                    f"from base class {base.__name__}")
                # if hasattr(attr_value, '__forceproperty__'):
                #     if attr_name not in namespace:
                #         raise ForceClassFormatError(f"Class {name} must implement property {attr_name} "
                #                                     f"from base class {base.__name__}")
        return super().__new__(mcs, name, bases, namespace)


# class TestA(metaclass=ForceFunctionMeta):
#     @forcefunction
#     def test(self):
#         pass
#
#
# class TestB(TestA):
#     pass
#     # def test(self):
#     #     pass
#
#
# if __name__ == '__main__':
#     TestB()
