import os
from yaml import load, SafeLoader

from .interfaces import SettingsProvider


class DemoSettingsProvider(SettingsProvider):
    """Settings provider for demo."""

    def get_settings(self) -> dict:
        """Get settings from environment or settings.yaml"""
        settings_file = (
            os.environ['SETTINGS_YAML']
            if 'SETTINGS_YAML' in os.environ
            else 'settings.yaml'
        )
        with open(settings_file, 'r') as f:
            return load(f, Loader=SafeLoader) or dict()
