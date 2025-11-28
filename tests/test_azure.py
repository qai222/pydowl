import os

import numpy as np
import pytest

from pydowl.data_type import _bytes_to_ndarray, _ndarray_to_bytes
from pydowl.sparql.settings import (
    ENV_AZURE_STORAGE_CONNECTION_STRING,
)
from pydowl.sparql.utils_azure import (
    blob_download_bytes_or_str,
    blob_upload_string_or_bytes,
)


@pytest.fixture(scope="module")
def test_blob_name():
    dummy_uuid = "fd3ef6f6-d14d-4c7d-a5a0-3ea20265ccc1"
    return dummy_uuid + "__LargeNode"


def test_blob_upload_bytes(test_blob_name):
    a = np.random.rand(555, 555)  # 2.35 mb uploaded
    blob_upload_string_or_bytes(
        connection_string=os.getenv(ENV_AZURE_STORAGE_CONNECTION_STRING),
        container_name="test-array",
        blob_name=test_blob_name,
        data=_ndarray_to_bytes(a),
    )


def test_blob_download_bytes(test_blob_name):
    a = np.random.rand(555, 555)  # 2.35 mb uploaded
    upload_info = blob_upload_string_or_bytes(
        connection_string=os.getenv(ENV_AZURE_STORAGE_CONNECTION_STRING),
        container_name="test-array",
        blob_name=test_blob_name,
        data=_ndarray_to_bytes(a),
    )
    storage_path = upload_info["storage_path"]
    # you need to define AZURE_TEST_CONTAINER_SAS_TOKEN in env
    downloaded = blob_download_bytes_or_str(
        storage_path, os.getenv("AZURE_TEST_CONTAINER_SAS_TOKEN")
    )
    a_rec = _bytes_to_ndarray(downloaded)
    assert np.allclose(a, a_rec)
