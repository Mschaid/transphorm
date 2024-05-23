import torch


def make_sinwaves(num_instances, num_points):
    x = torch.linspace(0, torch.pi, num_points)

    # Initialize an empty tensor to hold the sine waves
    sine_waves = torch.empty(num_instances, num_points)

    # Generate the sine waves
    for i in range(num_instances):
        frequency = i + 1  # Frequency ranges from 1 to num_instances
        sine_wave = torch.sin(frequency * x)
        sine_waves[i] = sine_wave
    return sine_waves


def make_sawtooths(num_instances, num_points):
    x = torch.linspace(0, 1, num_points)

    # Initialize an empty tensor to hold the sine waves
    saw_tooths = torch.empty(num_instances, num_points)

    # Generate the sine waves
    for i in range(num_instances):
        frequency = i + 1  # Frequency ranges from 1 to num_instances
        saw_tooth = (x * 10 * frequency) % 1
        saw_tooths[i] = saw_tooth
    return saw_tooths


def make_random_lines(num_instances, num_points):
    x = torch.linspace(0, 1, num_points)

    # Initialize an empty tensor to hold the random lines
    random_lines = torch.empty(num_instances, num_points)

    # Generate the random lines
    for i in range(num_instances):
        slope = torch.randn(1)[0] * 2 - 1  # Random slope
        intercept = torch.randn(1)[0] * 2 - 1  # Random intercept
        line = slope * x + intercept
        random_lines[i] = line
    return random_lines


def make_data():
    num_instances = 10000
    num_points = 1000
    saw_tooths = make_sawtooths(num_instances=num_instances, num_points=num_points)
    saw_tooth_lables = torch.zeros(num_instances, dtype=torch.float32)

    # sin_waves = make_sinwaves(num_instances=num_instances, num_points=num_points)
    random_lines = make_random_lines(num_instances=num_instances, num_points=num_points)
    random_lines_labels = torch.ones(num_instances, dtype=torch.float32)

    features = torch.concatenate((saw_tooths, random_lines))
    labels = torch.concatenate((saw_tooth_lables, random_lines_labels)).unsqueeze(1)
    return features, labels


""" Custom dataset objects"""


class TestTSDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        features, labels = make_data()
        features = features.view(-1, 1, 1000)
        self.data = TensorDataset(features, labels)

    def setup(self, stage):
        self.train, self.val, self.test = random_split(self.data, [0.7, 0.15, 0.15])

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
