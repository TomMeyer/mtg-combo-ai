# Copilot Docstring Instructions

## Preferred Docstring Format

Use the following Markdown-based docstring template for all functions and methods. Each section should have a `###` header, and arguments should be listed in bullet points. Defaults should be specified within backticks for readability.

### Template Example

# Function Docstring Format

Use the following format for function docstrings.

```python
def function_name(param1: Type, param2: Type = default) -> ReturnType:
    """
    Short description of the function’s purpose.

    ### Args
    - **param1** (Type):
        Description of the parameter.
    - **param2** (Type, optional):
        Description of the parameter. Defaults to `default`.

    ### Returns
    - ReturnType:
        Description of the return value.

    ### Raises
    - **ExceptionType**: Description of the error condition.
    - **AnotherExceptionType**: Description of the error condition.

    ### Example
    ```python
    result = function_name(param1_value, param2_value)
    ```
    """
```

# Class Docstring Format

Use the following format for class docstrings.

```python
class ClassName:
    """
    Short description of the class’s purpose.

    ### Attributes
    - **attribute1** (Type):
        Description of the attribute.
    - **attribute2** (Type, optional):
        Description of the attribute. Defaults to `default`.

    ### Methods
    - **method_name**: Brief description of what the method does.

    ### Example
    ```python
    instance = ClassName(attribute1_value)
    instance.method_name()
    ```
    """

    def __init__(self, attribute1: Type, attribute2: Type = default):
        """
        Initializes the class with given attributes.

        ### Args
        - **attribute1** (Type):
          Description of the attribute.
        - **attribute2** (Type, optional):
          Description of the attribute. Defaults to `default`.
        """
```


# Python Class Structure Template

Use the following class structure template for all Python classes. This template follows best practices for the order of elements within a class, ensuring consistency and readability.

### Recommended Order for Class Elements

1. **Class Docstring**: Briefly describe the class and its purpose.
2. **Class Attributes**: Any class-level constants or default values.
3. **`__init__` Method**: Constructor to initialize instance attributes.
4. **Class Methods (`@classmethod`)**: Alternative constructors or methods that need access to the class itself.
5. **Static Methods (`@staticmethod`)**: Utility methods that do not require access to the instance (`self`) or class (`cls`).
6. **Dunder (Magic) Methods**: Special methods like `__str__`, `__repr__`, or `__eq__`.
7. **Properties (`@property` and setters)**: Getter and setter methods for encapsulated attributes.
8. **Public Methods**: Methods intended for external use.
9. **Private Methods**: Helper methods intended for internal use only.

### Generic Python Class Template

```python
class ClassName:
    """
    Short description of the class’s purpose.
    """

    class_attribute = "default_value"

    def __init__(self, param1: Type, param2: Type):
        """
        Initializes the class with given parameters.

        Args:
            param1 (Type): Description of the first parameter.
            param2 (Type): Description of the second parameter.
        """
        self._param1 = param1
        self._param2 = param2

    @classmethod
    def from_alternative(cls, param: Type):
        """
        Alternative constructor using a different parameter.

        Args:
            param (Type): Description of the parameter.

        Returns:
            ClassName: An instance of the class.
        """
        return cls(param, "default_value")

    @staticmethod
    def utility_method(param: Type) -> ReturnType:
        """
        A static utility method.

        Args:
            param (Type): Description of the parameter.

        Returns:
            ReturnType: Description of the return value.
        """
        return some_computation(param)

    def __str__(self) -> str:
        """Returns a string representation of the instance."""
        return f"ClassName(param1={self._param1}, param2={self._param2})"

    @property
    def param1(self) -> Type:
        """Gets the value of param1."""
        return self._param1

    @param1.setter
    def param1(self, value: Type):
        """Sets the value of param1."""
        self._param1 = value

    def public_method(self, arg: Type) -> ReturnType:
        """
        A public method description.

        Args:
            arg (Type): Description of the argument.

        Returns:
            ReturnType: Description of the return value.
        """
        return some_computation(arg)

    def _private_method(self) -> ReturnType:
        """
        A private method for internal use only.

        Returns:
            ReturnType: Description of the internal computation.
        """
        return self._param1 * 42
