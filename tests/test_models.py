from hh_goa_rag.models import (
    JINA_CODE_REVISION,
    MODEL_FILE_PATTERNS,
    MODEL_SPECS,
    safe_model_name,
)


def test_model_download_patterns_include_common_weight_formats() -> None:
    assert "*.safetensors" in MODEL_FILE_PATTERNS
    assert "*.bin" in MODEL_FILE_PATTERNS


def test_safe_model_name_stays_within_one_directory() -> None:
    assert safe_model_name("owner/model") == "owner__model"


def test_gte_compatibility_workaround_is_explicit() -> None:
    spec = MODEL_SPECS["Alibaba-NLP/gte-multilingual-base"]
    assert spec.reset_position_ids_buffer
    assert spec.adapter_version


def test_jina_secondary_code_is_immutable() -> None:
    assert len(JINA_CODE_REVISION) == 40
