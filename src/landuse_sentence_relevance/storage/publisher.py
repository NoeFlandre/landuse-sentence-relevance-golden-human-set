from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from landuse_sentence_relevance.domain.constraints import validate_final_dataset
from landuse_sentence_relevance.domain.models import Annotation
from landuse_sentence_relevance.domain.selection import select_final_annotations


class DatasetUploader(Protocol):
    """Upload a prepared public dataset to the configured Hub repository."""

    def __call__(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]],
        token: str | None,
        private: bool,
    ) -> None: ...


class DatasetPublisher:
    """Publish the final balanced subset without an extra confirmation step."""

    def __init__(
        self,
        dataset_id: str,
        token: str | None = None,
        uploader: DatasetUploader | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        self._dataset_id = dataset_id
        self._token = token
        self._uploader = uploader or self._upload_to_hub
        self._cleanup = cleanup

    def publish_if_ready(self, annotations: Iterable[Annotation]) -> bool:
        selected = select_final_annotations(annotations)
        if selected is None:
            return False
        validate_final_dataset(selected)
        self._uploader(
            dataset_id=self._dataset_id,
            records=[annotation.to_dict() for annotation in selected],
            token=self._token,
            private=False,
        )
        if self._cleanup is not None:
            self._cleanup()
        return True

    def _upload_to_hub(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]],
        token: str | None,
        private: bool,
    ) -> None:
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi
        except ImportError as error:  # pragma: no cover - optional publishing environment
            raise RuntimeError(
                "Install the project dependencies with `uv sync` to publish the dataset"
            ) from error

        api = HfApi(token=token)
        api.create_repo(repo_id=dataset_id, repo_type="dataset", private=private, exist_ok=True)
        Dataset.from_list(records).push_to_hub(
            dataset_id,
            split="train",
            token=token,
            commit_message="Publish balanced human annotations",
        )
