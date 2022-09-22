from math import log, e

def subtract_vectors(v, w):
    """
    Returns the result of element-wise vector subtraction between v and w; v-w
    """
    if len(v) != len(w):
        raise ValueError('Vectors are incompatible')

    return [v[i] - w[i] for i in range(len(v))]


def dot_product(v, w):
    """
    Returns dot product of two vectors v and w
    """
    if len(v) != len(w):
        raise ValueError('Vectors are incompatible')
        
    return sum([v[i] * w[i] for i in range(len(v))])

def sigmoid(z):
    """
    Sigmoid function that prevents overflow
    """
    return 1 / (1 + e**-z) if z >= 0 else e**z / (1 + e**z)

class LogisticRegression:
    def __init__(self, X, y, eta=.05, alpha=.1, max_iter=25, max_loss=.2, threshold=.5):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.eta = eta
        self.w = [0] * len(X[0])
        self.b = 0
        self.max_iter = max_iter
        self.max_loss = max_loss
        self.threshold = threshold
        self._optimize_loss()

    def _optimize_loss(self) -> None:
        """
        Finds weights and bias that minimize loss
        """
        loss, prev_loss, i = 1, 1, 0
        predictions = [0] * len(self.y)
        while self.max_loss < abs(loss) and i < self.max_iter:
            self.gradient_descent(predictions)
            predictions = self.get_likelihoods()
            loss = self.calculate_loss(predictions)
            i += 1

        if i > self.max_iter:
            print(f'Regression was unable to converge within {self.max_iter} iterations.')

    def gradient_descent(self, predictions) -> None:
        """
        w(t+1) = w(t) - n((y_hat_j - y_j)*x_j)
        b(t+1) = b(t) - n((y_hat_j - y_j))
        𝜃𝑗:=𝜃𝑗(1−𝛼𝜆𝑚)(𝛼𝑚 ∑𝑖=1𝑚(ℎ𝜃(𝑥(𝑖))−𝑦(𝑖))𝑥(𝑖)𝑗)
        """
        
        scaled_error = list(map(lambda x: x * self.eta, subtract_vectors(predictions, self.y)))
        for i, row in enumerate(self.X):
            self.b -= scaled_error[i]  # update bias
            self.w = [self.w[j] - scaled_error[i] * row[j] for j in range(len(self.w))]  # update weights

    def get_likelihoods(self):
        """
        :return: the likelihood that each observation is in a positive class
        """
        return [sigmoid(dot_product(self.w, obs) + self.b) for obs in self.X]

    def calculate_loss(self, likelihoods):
        """
        :param likelihoods: a list of the current likelihood for each observation being in the positive class
        :return: the average cross entropy loss over all estimates
        loss = (1-y_hat)*log(1-p(y_i)) - y_hat*log(p(y_i))
        """
        
        losses = [(1 - self.y[i]) * log(1 - sigmoid(likelihoods[i])) - self.y[i] * log(sigmoid(likelihoods[i]))
                  for i in range(len(likelihoods))]

        return -(sum(losses) ** 2 / len(losses))

    def predict(self, observations):
        """
        :param observations: 2d list of observations with values for each predictor
        :return: the class predicted by the trained logistic regression
        """
        likelihoods = [sigmoid(dot_product(self.w, obs) + self.b) for obs in observations]
        return [1 if likelihood >= self.threshold else 0 for likelihood in likelihoods]

    def classification_error(self, observations, outcomes):
        """
        :param observations: list of observations you want to predict
        :param outcomes: list of correct classifications for each observation
        :return: the classification erorr from the model's prediction
        """
        predictions = self.predict(observations)
        return sum([predictions[i] == outcomes[i] for i in range(len(observations))]) / len(observations)
