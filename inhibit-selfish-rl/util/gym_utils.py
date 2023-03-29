"""
Author: cemiu (Philipp B.)
"""

import os


class TrainingFiles:
    """Helper class for creating training files."""
    # log_path: training/logs
    # model_path: training/models/<model_name>
    def __init__(
            self,
            model_name: str,
            base_path: str = '.',
            log_path: str = 'training/logs',
            model_path: str = 'training/models'
    ):
        if '/' in model_name:
            model_name = model_name.replace('/', '-')
        self._log_path = os.path.join(base_path, log_path, model_name).replace('\\', '/')
        self._model_path = os.path.join(base_path, model_path, model_name).replace('\\', '/')
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

    @property
    def log(self):
        return self._log_path

    @property
    def model(self):
        """Return model path with or without .zip extension."""
        return self._model_path

    @property
    def model_zip(self):
        return self.model + '.zip'
