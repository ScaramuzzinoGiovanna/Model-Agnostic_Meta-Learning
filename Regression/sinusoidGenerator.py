import numpy as np


class SinusoidGenerator():
    def __init__(self, K=10):
        self.K = K

    def generate_amplitude_phase(self):
        self.amplitude = np.random.uniform(0.1, 5)
        self.phase = np.random.uniform(0, np.pi)
        return self.amplitude, self.phase

    def generate_datapoints(self):
        # K datapoints
        return np.random.uniform(-5, 5, self.K)

    def sin_function(self, x):
        return self.amplitude * np.sin(x + self.phase)

    def generate_equallySpacedPoints(self, n_points):
        # punti per disegnare la funzione
        x_tab = np.linspace(-5, 5, n_points)
        y_tab = self.sin_function(x_tab)
        return x_tab, y_tab

    def create_dataset(self, n_task, test=False):
        dataset_x = []
        dataset_y = []
        tabular_x = []
        tabular_y = []
        for _ in range(n_task):  # genero ntask esempi di seno
            self.generate_amplitude_phase()
            x = self.generate_datapoints()
            y = self.sin_function(x)
            dataset_x.append(x)
            dataset_y.append(y)
            if test:
                x_tab, y_tab = self.generate_equallySpacedPoints(100)
                tabular_x.append(x_tab)
                tabular_y.append(y_tab)
        if test:
            return [dataset_x, dataset_y], tabular_x, tabular_y
        else:
            return [dataset_x, dataset_y]

    def create_dataset_with_specific_point(self, x):
        # dati K punti x genero una funzione seno casuale e i punti xtab
        x = np.array(x)
        self.generate_amplitude_phase()
        y = self.sin_function(x)
        x_tab, y_tab = self.generate_equallySpacedPoints(100)
        return [[x], [y]], [x_tab], [y_tab]
