import numpy as np


class Perceptron(object):

    def __init__(
        self, n_inputs: int, max_epochs: int = 100, learning_rate: float = 0.01
    ) -> None:
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs: int = n_inputs
        self.max_epochs: int = max_epochs
        self.learning_rate: float = learning_rate
        self.weights: np.ndarray = np.zeros(n_inputs + 1)

    def forward(self, input_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            np.array: Predicted label (1 or -1) or Predicted labels.
        """
        input_vec_with_bias = np.insert(input_vec, 0, 1, axis=1)
        res_mat = np.dot(input_vec_with_bias, self.weights)
        labels = np.where(res_mat > 0, 1, -1)
        return res_mat, labels

    def train(
        self,
        training_inputs: np.ndarray,
        labels: np.ndarray,
        test_inputs: np.ndarray,
        test_labels: np.ndarray,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        np.random.seed(42)
        train_acc_list: list[float] = []
        train_loss_list: list[float] = []
        test_acc_list: list[float] = []
        test_loss_list: list[float] = []
        # we need max_epochs to train our model
        for times in range(self.max_epochs):
            """
            What we should do in one epoch ?
            you are required to write code for
            1.do forward pass
            2.calculate the error
            3.compute parameters' gradient
            4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
            please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            total = training_inputs.shape[0]
            training_inputs = np.array(training_inputs)
            labels = np.array(labels)

            pred_loc, pred_labels = self.forward(training_inputs)
            train_acc, train_loss = self.get_acc_loss(pred_loc, pred_labels, labels)
            for i in range(total):
                label, prediction = labels[i], pred_labels[i]
                example = np.insert(training_inputs[i], 0, 1)
                if label * prediction < 0:
                    self.weights += self.learning_rate * (label * example)

            test_pred_loc, test_pred_label = self.forward(test_inputs)
            test_acc, test_loss = self.get_acc_loss(
                test_pred_loc, test_pred_label, test_labels
            )

            train_acc_list.append(train_acc * 100)
            train_loss_list.append(train_loss)
            test_acc_list.append(test_acc * 100)
            test_loss_list.append(test_loss)

            if times == self.max_epochs - 1:
                print(f"Final test loss:{test_loss:.6f}, acc: {100 * test_acc:.2f}%")

        return train_acc_list, train_loss_list, test_acc_list, test_loss_list

    def get_acc_loss(
        self, pred_loc: np.ndarray, pred_labels: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        right_num = np.sum(labels == pred_labels)
        total = pred_labels.shape[0]
        acc = right_num / total
        losses = np.maximum(0, -(labels * pred_loc))
        loss = np.sum(losses) / total
        return acc.item(), loss.item()
