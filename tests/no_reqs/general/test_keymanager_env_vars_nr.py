import os
import pytest

from skinai.config import Settings
from fastapi.responses import JSONResponse
import json
from skinai.api.dependencies.common import get_user_manager
from skinai.api.services.user import UserManager
from weaviate.util import generate_uuid5

from uuid import uuid4

from skinai import tool
from skinai.tree import Tree
from skinai.config import SkinAIKeyManager


def test_api_keys_are_not_passed_down():

    os.environ["TEST_API_KEY"] = "1234567890"
    settings = Settings()
    with SkinAIKeyManager(settings):
        assert "TEST_API_KEY" not in os.environ


def test_modified_api_keys_are_passed_down():

    os.environ["TEST_API_KEY_"] = "1234567890"
    settings = Settings()
    with SkinAIKeyManager(settings):
        assert "TEST_API_KEY_" in os.environ

    settings = Settings()
    settings.configure(api_keys={"TEST_API_KEY": "1234567890"})
    with SkinAIKeyManager(settings):
        assert "TEST_API_KEY" in os.environ
