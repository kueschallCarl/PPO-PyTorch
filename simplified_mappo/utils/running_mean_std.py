import numpy as np

class RunningMeanStd:
    """
    Tracks the running mean and standard deviation of a data stream.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, epsilon=1e-4, shape=()):
        """
        Initialize running mean and standard deviation.
        
        Args:
            epsilon (float): Small constant for numerical stability
            shape (tuple): Shape of the data to track
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.eps = epsilon

    def update(self, x):
        """
        Update running mean and variance with new data using Welford's online algorithm.
        
        Args:
            x: New data point(s) to include in the running statistics
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """
        Normalize data using the running mean and standard deviation.
        
        Args:
            x: Data to normalize
            
        Returns:
            Normalized data with zero mean and unit variance
        """
        return (x - self.mean) / np.sqrt(self.var + self.eps)

    @property
    def std(self):
        """
        Get the current standard deviation.
        
        Returns:
            Standard deviation as numpy array
        """
        return np.sqrt(self.var + self.eps)

    def reset(self):
        """Reset statistics to initial state."""
        self.__init__(epsilon=self.eps)
