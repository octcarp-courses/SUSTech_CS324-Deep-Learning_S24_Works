import csv
import numpy as np

# Default constants
EXP_DATA: str = "./result/latest.csv"

DNN_HIDDEN_UNITS_DEFAULT: str = "20"
LEARNING_RATE_DEFAULT: float = 1e-2
MAX_EPOCHS_DEFAULT: int = 1500  # adjust if you use batch or not
EVAL_FREQ_DEFAULT: int = 10
BATCH_SIZE_DEFAULT: int = 800
RANDOM_SEED_DEFAULT: int = 42


class Layer:
    def __init__(self) -> None: ...

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass must implement abstract method 'forward'")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass must implement abstract method 'backward'")


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int) -> None:
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


class NumpyMLP:
    def __init__(self, n_inputs: int, n_hidden: list[int], n_classes: int) -> None:
        """
        Initializes the multi-layer perceptron object.

        This function should initialize the layers of the MLP including any linear layers and activation functions
        you plan to use. You will need to create a list of linear layers based on n_inputs, n_hidden, and n_classes.
        Also, initialize ReLU activation layers for each hidden layer and a softmax layer for the output.

        Args:
            n_inputs (int): Number of inputs (i.e., dimension of an input vector).
            n_hidden (list of int): List of integers, where each integer is the number of units in each hidden layer.
            n_classes (int): Number of classes of the classification problem (i.e., output dimension of the network).
        """
        # Hint: You can use a loop to create the necessary number of layers and add them to a list.
        # Remember to initialize the weights and biases in each layer.
        self.layers: list[Layer] = []
        prev_size = n_inputs
        for unit_size in n_hidden:
            self.layers.append(Linear(prev_size, unit_size))
            self.layers.append(ReLU())
            prev_size = unit_size
        self.layers.append(Linear(prev_size, n_classes))
        self.layers.append(SoftMax())

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the network output from the input by passing it through several layers.

        Here, you should implement the forward pass through all layers of the MLP. This involves
        iterating over your list of layers and passing the input through each one sequentially.
        Don't forget to apply the activation function after each linear layer except for the output layer.

        Args:
            x (numpy.ndarray): Input to the network.

        Returns:
            numpy.ndarray: Output of the network.
        """
        # Start with the input as the initial output
        out = x

        # Implement the forward pass through each layer.
        # Hint: For each layer in your network, you will need to update 'out' to be the layer's output.
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, dout: np.ndarray) -> None:
        """
        Performs the backward propagation pass given the loss gradients.

        Here, you should implement the backward pass through all layers of the MLP. This involves
        iterating over your list of layers in reverse and passing the gradient through each one sequentially.
        You will update the gradients for each layer.

        Args:
            dout (numpy.ndarray): Gradients of the loss with respect to the output of the network.
        """
        # Implement the backward pass through each layer.
        # Hint: You will need to update 'dout' to be the gradient of the loss with respect to the input of each layer.

        # No need to return anything since the gradients are stored in the layers.

        for layer in reversed(self.layers):
            dout = layer.backward(dout)


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.

    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding

    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    predicted_classes = np.argmax(predictions, axis=1)
    targets_classes = np.argmax(targets, axis=1)
    correct = np.sum(predicted_classes == targets_classes)
    acc = correct / predictions.shape[0]
    return acc.item()


def train(
    data,
    batch_size: int = BATCH_SIZE_DEFAULT,
    dnn_hidden_units: str = DNN_HIDDEN_UNITS_DEFAULT,
    learning_rate: float = LEARNING_RATE_DEFAULT,
    max_steps: int = MAX_EPOCHS_DEFAULT,
    eval_freq: int = EVAL_FREQ_DEFAULT,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """
    Performs training and evaluation of MLP model.

    Args:
        data: training data
        batch_size: Int for batch size
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set

    NOTE: Add necessary arguments such as the data, your model...
    """
    np.random.seed(random_seed)

    # Load your data here
    train_x, train_y, test_x, test_y = data

    # Initialize your MLP model and loss function (CrossEntropy) here
    dnn_hidden_units_list = [int(x) for x in dnn_hidden_units.split(",")]
    mlp: NumpyMLP = NumpyMLP(n_inputs=2, n_hidden=dnn_hidden_units_list, n_classes=2)
    loss_fn: CrossEntropy = CrossEntropy()

    train_length: int = train_x.shape[0].item()
    eval_step_l: list[int] = []
    train_acc_l: list[float] = []
    test_acc_l: list[float] = []
    train_loss_l: list[float] = []
    test_loss_l: list[float] = []
    batch_sz: int = np.minimum(batch_size, train_length).item()
    batch_sz = np.maximum(batch_sz, -train_length).item()
    trainable_layers: list[Linear] = [
        layer for layer in mlp.layers if isinstance(layer, Linear)
    ]

    for step in range(max_steps):
        train_acc: float = 0.0
        train_loss: float = 0.0
        random_index: np.ndarray = np.random.permutation(train_length)
        shuffled_x, shuffled_y = train_x[random_index], train_y[random_index]
        if batch_sz < 0:
            actual_batch = -batch_sz
            index = np.random.randint(low=0, high=train_length - actual_batch + 1)
            step_x = shuffled_x[index : index + actual_batch]
            step_y = shuffled_y[index : index + actual_batch]
            predictions = mlp.forward(step_x)
            loss = loss_fn.backward(predictions, step_y)
            mlp.backward(loss)
            train_acc = accuracy(predictions, step_y)
            # train_loss = loss_fn.forward(predictions, step_y)
            for layer in trainable_layers:
                layer.update_params(learning_rate)
        else:
            lr = learning_rate / batch_size
            for start_idx in range(0, train_length - batch_sz + 1, batch_sz):
                step_x = shuffled_x[start_idx : start_idx + batch_sz]
                step_y = shuffled_y[start_idx : start_idx + batch_sz]
                predictions = mlp.forward(step_x)
                loss = loss_fn.backward(predictions, step_y)
                mlp.backward(loss)
                train_acc += accuracy(predictions, step_y)
                train_loss += loss_fn.forward(predictions, step_y)

                for layer in trainable_layers:
                    layer.update_params(learning_rate=lr)

            iter_cnt = train_length // batch_sz
            train_acc /= iter_cnt
            train_loss /= iter_cnt

        if step % eval_freq == 0 or step == max_steps - 1:
            test_predictions = mlp.forward(test_x)
            test_acc = accuracy(test_predictions, test_y)
            test_loss = loss_fn.forward(test_predictions, test_y).item()
            # # print(f"Step: {step}, Loss: {test_loss:.6f}, Accuracy: {test_acc:.2f}%")

            eval_step_l.append(step)
            train_acc_l.append(train_acc)
            test_acc_l.append(test_acc)
            train_loss_l.append(train_loss)
            test_loss_l.append(test_loss)

    return eval_step_l, train_acc_l, test_acc_l, train_loss_l, test_loss_l
