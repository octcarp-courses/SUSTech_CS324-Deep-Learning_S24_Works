import numpy as np


class Layer:
    def __init__(self) -> None: ...

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass must implement abstract method 'forward'")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass must implement abstract method 'backward'")


class Linear(Layer):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer.
        Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self._params: dict[str, np.ndarray] = {
            "weight": np.random.normal(
                loc=0, scale=0.1, size=(in_features, out_features)
            ),
            "bias": np.zeros(shape=out_features),
        }
        self._grads: dict[str, np.ndarray] = {
            "weight": np.zeros(shape=(in_features, out_features)),
            "bias": np.zeros(shape=out_features),
        }
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass using the formula: output = xW + b
        Implement the forward pass.
        """
        self._x = x
        res = np.dot(x, self._params["weight"]) + self._params["bias"]
        return res

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        Implement the backward pass.
        """
        if self._x is None:
            raise RuntimeError("Forward pass must be called before backward pass.")
        self._grads["weight"] = np.dot(self._x.T, dout)
        self._grads["bias"] = np.sum(dout, axis=0)
        d_loss = np.dot(dout, self._params["weight"].T)
        return d_loss

    def update_params(self, learning_rate: float, clear: bool = True) -> None:
        self._params["weight"] -= learning_rate * self._grads["weight"]
        self._params["bias"] -= learning_rate * self._grads["bias"]
        if clear:
            self._grads["weight"].fill(0)
            self._grads["bias"].fill(0)


class ReLU(Layer):
    def __init__(self) -> None:
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        Implement the forward pass.
        """
        self._x = x
        res = np.maximum(0, x)
        return res

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the ReLU function.
        Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        if self._x is None:
            raise RuntimeError("Forward pass must be called before backward pass.")
        d_loss = dout
        d_loss[self._x < 0] = 0
        return d_loss


class SoftMax(Layer):
    def __init__(self) -> None:
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        Implement the forward pass using the Max Trick for numerical stability.
        """
        self._x = x
        x_max = np.max(x, axis=1, keepdims=True)
        y = np.exp(x - x_max)
        res = y / np.sum(y, axis=1, keepdims=True)
        return res

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout


class CrossEntropy:
    def __init__(self) -> None:
        self._DELTA: float = 1e-7

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        Implement the forward pass.
        """
        if x.shape != y.shape:
            raise ValueError("Input shapes for x and y must match.")
        res = -np.sum(y * np.log(x + self._DELTA)) / x.shape[0]
        return res

    def backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        d_loss = x + self._DELTA - y
        return d_loss
