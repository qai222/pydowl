from azure.core.exceptions import ResourceExistsError, AzureError
from azure.storage.blob import BlobClient
from azure.storage.blob import BlobServiceClient
from loguru import logger


def blob_upload_string_or_bytes(
    connection_string: str, container_name: str, blob_name: str, data: bytes | str
):
    """
    Upload a string or bytes payload to Azure Blob Storage, overwriting any
    existing blob with the same name.

    The function is intentionally liberal about ``data``: if a ``str`` is
    provided it is passed straight to the SDK (which will UTF-8 encode it);
    ``bytes`` are uploaded as-is. Containers are created on demand using the
    provided connection string, subject to Azure's DNS-style naming rules
    (lowercase letters/numbers, hyphens between alphanumerics, length 3–63).

    Args:
        connection_string (str): Azure Storage connection string.
        container_name (str): Name of the container to upload the blob to.
        blob_name (str): Name of the blob.
        data (bytes | str): Payload to upload.
    """
    try:
        # Create a BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Create the container if it doesn't exist
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
            logger.debug(f"Container '{container_name}' created.")
        except ResourceExistsError:
            logger.warning(f"Container '{container_name}' already exists.")

        # Create a BlobClient
        blob_client = container_client.get_blob_client(blob_name)

        # Upload the byte data
        upload_info = blob_client.upload_blob(data, overwrite=True)
        logger.debug(
            f"Data uploaded to blob '{blob_name}' in container '{container_name}'."
        )

        return {
            "blob_name": blob_name,
            "container_name": container_name,
            "storage_path": f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}",
            "upload_info": upload_info,
        }

    except AzureError as e:
        logger.error(f"Failed to upload data to Azure Blob Storage: {e}")
        raise e

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e


def blob_download_bytes_or_str(blob_url: str, sas_token: str) -> bytes:
    """
    Download a blob's raw bytes from Azure Blob Storage.

    ``sas_token`` is appended only if provided (empty/whitespace strings are
    ignored). Callers are responsible for decoding the returned bytes into
    text or structured data as appropriate.

    Args:
        blob_url (str): Full blob URL (may already include a SAS token).
        sas_token (str): SAS token to use when ``blob_url`` does not embed one.

    Returns:
        bytes: The blob contents.

    Raises:
        AzureError: If there is an issue accessing or downloading the blob.
        Exception: For any other unexpected errors.
    """
    try:
        token = (sas_token or "").strip().lstrip("?")
        blob_client = BlobClient.from_blob_url(
            blob_url, credential=token if token else None
        )
        return blob_client.download_blob().readall()
        # blob_client = BlobClient.from_blob_url(blob_url + f"?{sas_token}")
        # # Download the blob's content
        # download_stream = blob_client.download_blob()
        # blob_content = download_stream.readall()
        # return blob_content

    except AzureError as e:
        logger.error(f"Azure error occurred while downloading the blob: {e}")
        raise e

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e
