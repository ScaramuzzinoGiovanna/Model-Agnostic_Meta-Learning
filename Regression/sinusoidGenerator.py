import numpy as np


class SinusoidGenerator():
    def __init__(self, K=10):
        self.K = K  # number of datapoints

    def generate_amplitude_phase(self):
        self.amplitude = np.random.uniform(0.1, 5.0)
        self.phase = np.random.uniform(0, np.pi)
        return self.amplitude, self.phase

    def generate_datapoints(self):
        # generates K datapoints
        return np.random.uniform(-5, 5, self.K)

    def sin_function(self, x):
        return self.amplitude * np.sin(x + self.phase)

    def generate_equallySpacedPoints(self, n_points):
        # generates points to draw the function
        x_tab = np.linspace(-5, 5, n_points)
        y_tab = self.sin_function(x_tab)
        return x_tab, y_tab

    def create_dataset(self, n_task, test=False):
        dataset_x = []
        dataset_y = []
        tabular_x = []
        tabular_y = []
        for _ in range(n_task):  # generate n_task examples of sine functions
            self.generate_amplitude_phase()
            x = self.generate_datapoints()
            y = self.sin_function(x)
            dataset_x.append(x[:, None])
            dataset_y.append(y[:, None])
            if test:
                x_tab, y_tab = self.generate_equallySpacedPoints(100)
                tabular_x.append(x_tab[:, None])
                tabular_y.append(y_tab[:, None])
        if test:
            return [dataset_x, dataset_y], tabular_x, tabular_y
        else:
            return [dataset_x, dataset_y]

    def create_dataset_with_specific_point(self, x):
        # given x (= K points) generates a random sine function and xtab points
        x = np.array(x)
        self.generate_amplitude_phase()
        y = self.sin_function(x)
        x_tab, y_tab = self.generate_equallySpacedPoints(100)
        return [[x[:, None]], [y[:, None]]], [x_tab[:, None]], [y_tab[:, None]]


if __name__ == "__main__":
    sin_gen = SinusoidGenerator(5)
    dataset_train = sin_gen.create_dataset(5)
    dataset_test, x_tab, y_tab = sin_gen.create_dataset(9, test=True)
    for i in range(len(dataset_train[0])):
        x = dataset_train[0][i]
        y = dataset_train[1][i]
        print('x', x)
        print('y', y)
