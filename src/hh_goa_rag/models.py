"""Project-local embedding model acquisition and inference adapters."""

from __future__ import annotations

import gc
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from sentence_transformers import SentenceTransformer
from torch.nn import functional as torch_functional
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from hh_goa_rag.io import write_json
from hh_goa_rag.metrics import latency_percentiles


@dataclass(frozen=True)
class ModelSpec:
    repository: str
    query_prefix: str = ""
    passage_prefix: str = ""
    query_task: str | None = None
    passage_task: str | None = None
    mean_pooling_base_encoder: bool = False
    reset_position_ids_buffer: bool = False
    adapter_version: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "BAAI/bge-m3": ModelSpec("BAAI/bge-m3"),
    "intfloat/multilingual-e5-small": ModelSpec(
        "intfloat/multilingual-e5-small", query_prefix="query: ", passage_prefix="passage: "
    ),
    "l3cube-pune/indic-sentence-bert-nli": ModelSpec(
        "l3cube-pune/indic-sentence-bert-nli"
    ),
    "l3cube-pune/indic-sentence-similarity-sbert": ModelSpec(
        "l3cube-pune/indic-sentence-similarity-sbert"
    ),
    "sentence-transformers/paraphrase-MiniLM-L3-v2": ModelSpec(
        "sentence-transformers/paraphrase-MiniLM-L3-v2"
    ),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": ModelSpec(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ),
    "intfloat/multilingual-e5-base": ModelSpec(
        "intfloat/multilingual-e5-base", query_prefix="query: ", passage_prefix="passage: "
    ),
    "Alibaba-NLP/gte-multilingual-base": ModelSpec(
        "Alibaba-NLP/gte-multilingual-base",
        reset_position_ids_buffer=True,
        adapter_version="transformers5-reset-nonpersistent-buffers-v3",
    ),
    "jinaai/jina-embeddings-v3": ModelSpec(
        "jinaai/jina-embeddings-v3",
        query_task="retrieval.query",
        passage_task="retrieval.passage",
        adapter_version="transformers5-local-code-v2-fix-tokenizer-regex",
    ),
    "ai4bharat/IndicBERT-v3-4B": ModelSpec(
        "ai4bharat/IndicBERT-v3-4B", mean_pooling_base_encoder=True
    ),
}

MODEL_FILE_PATTERNS = [
    "*.json",
    "*.py",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.model",
    "*.txt",
    "*.tiktoken",
]
JINA_CODE_REPOSITORY = "jinaai/xlm-roberta-flash-implementation"
JINA_CODE_REVISION = "845308d0fd72a8406a3e378450e1a09522790419"


def safe_model_name(repository: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", repository)


def acquire_model(repository: str, model_root: str | Path) -> tuple[Path, str]:
    """Download a model to an exact project-local, revision-addressed directory."""
    revision = HfApi().model_info(repository).sha
    target = Path(model_root) / f"{safe_model_name(repository)}--{revision[:12]}"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=target,
        allow_patterns=MODEL_FILE_PATTERNS,
    )
    additional_code: dict[str, str] | None = None
    if repository == "jinaai/jina-embeddings-v3":
        snapshot_download(
            repo_id=JINA_CODE_REPOSITORY,
            revision=JINA_CODE_REVISION,
            local_dir=target,
            allow_patterns=["*.py"],
        )
        lora_path = target / "modeling_lora.py"
        lora_source = lora_path.read_text(encoding="utf-8")
        lifecycle_anchor = (
            "        self.main_params_trainable = config.lora_main_params_trainable\n"
        )
        lifecycle_patch = (
            lifecycle_anchor + "        self.post_init()  # Transformers >=5 lifecycle\n"
        )
        if lifecycle_patch not in lora_source:
            if lora_source.count(lifecycle_anchor) != 1:
                raise RuntimeError("Jina compatibility patch anchor changed upstream")
            lora_path.write_text(
                lora_source.replace(lifecycle_anchor, lifecycle_patch), encoding="utf-8"
            )
        config_path = target / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["auto_map"] = {
            key: value.split("--", maxsplit=1)[-1] for key, value in config["auto_map"].items()
        }
        write_json(config_path, config)
        additional_code = {
            "repository": JINA_CODE_REPOSITORY,
            "revision": JINA_CODE_REVISION,
            "compatibility_patch": "Call post_init on XLMRobertaLoRA for Transformers >=5",
        }
    python_hashes = {
        str(path.relative_to(target)).replace("\\", "/"): _sha256_file(path)
        for path in target.rglob("*.py")
        if path.is_file()
    }
    write_json(
        target / ".hh_goa_model.json",
        {
            "repository": repository,
            "revision": revision,
            "additional_code": additional_code,
            "python_file_sha256": python_hashes,
            "owned_by": "hh-goa-retrieval-ablation",
        },
    )
    return target, revision


class EmbeddingModel:
    """SentenceTransformers adapter with model-card-required asymmetric inputs."""

    def __init__(
        self,
        spec: ModelSpec,
        path: Path,
        *,
        device: str,
        max_sequence_length: int,
        dtype: str,
    ) -> None:
        _verify_model_artifact(path, spec.repository)
        self.spec = spec
        self.device = device
        self.max_sequence_length = max_sequence_length
        torch_dtype = getattr(torch, dtype)
        if spec.mean_pooling_base_encoder:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(path), trust_remote_code=True, model_max_length=max_sequence_length
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(path),
                trust_remote_code=True,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            self.model.eval()
            return
        if spec.repository == "jinaai/jina-embeddings-v3":
            self._prepare_jina_dynamic_cache(path)
        tokenizer_kwargs: dict[str, Any] = {"model_max_length": max_sequence_length}
        if spec.repository == "jinaai/jina-embeddings-v3":
            tokenizer_kwargs["fix_mistral_regex"] = True
        self.model = SentenceTransformer(
            str(path),
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch_dtype},
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.model.max_seq_length = max_sequence_length
        if spec.reset_position_ids_buffer:
            # Transformers 5 meta-device loading leaves this non-persistent buffer
            # uninitialized in Alibaba's custom implementation. Restore the exact
            # arange initialization declared by NewEmbeddings.__init__.
            transformer = self.model[0].auto_model
            embeddings = transformer.embeddings
            position_ids = torch.arange(
                transformer.config.max_position_embeddings,
                device=embeddings.word_embeddings.weight.device,
            )
            embeddings.register_buffer("position_ids", position_ids, persistent=False)
            old_rotary = embeddings.rotary_emb
            rotary_kwargs: dict[str, Any] = {
                "dim": old_rotary.dim,
                "max_position_embeddings": old_rotary.max_position_embeddings,
                "base": old_rotary.base,
                "device": embeddings.word_embeddings.weight.device,
            }
            if hasattr(old_rotary, "scaling_factor"):
                rotary_kwargs["scaling_factor"] = old_rotary.scaling_factor
                rotary_kwargs["mixed_b"] = old_rotary.mixed_b
            embeddings.rotary_emb = type(old_rotary)(**rotary_kwargs)

    @staticmethod
    def _prepare_jina_dynamic_cache(path: Path) -> None:
        """Work around Transformers 5 not copying nested local relative imports."""
        import transformers.dynamic_module_utils as dynamic_modules

        default_cache = Path(dynamic_modules.HF_MODULES_CACHE)
        module_cache = path.parent.parent / "remote_modules"
        dynamic_modules.HF_MODULES_CACHE = str(module_cache.resolve())
        submodule = dynamic_modules._sanitize_module_name(path.name)
        for cache_root in {module_cache, default_cache}:
            destination = cache_root / dynamic_modules.TRANSFORMERS_DYNAMIC_MODULE_NAME / submodule
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "__init__.py").touch(exist_ok=True)
            for source in path.glob("*.py"):
                shutil.copy2(source, destination / source.name)

    def _encode(
        self,
        texts: list[str],
        *,
        kind: str,
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        if self.spec.mean_pooling_base_encoder:
            return self._encode_indic(texts, batch_size=batch_size, show_progress=show_progress)
        prefix = self.spec.query_prefix if kind == "query" else self.spec.passage_prefix
        task = self.spec.query_task if kind == "query" else self.spec.passage_task
        prepared = [prefix + text for text in texts]
        kwargs: dict[str, Any] = {}
        if task:
            kwargs.update({"task": task, "prompt_name": task})
        result = self.model.encode(
            prepared,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            **kwargs,
        )
        vectors = np.asarray(result, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def _encode_indic(
        self, texts: list[str], *, batch_size: int, show_progress: bool
    ) -> np.ndarray:
        """Mean-pool the bidirectional text backbone without materializing LM logits."""
        vectors: list[np.ndarray] = []
        starts = range(0, len(texts), batch_size)
        batches = tqdm(starts, desc="IndicBERT batches", disable=not show_progress)
        for start in batches:
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                output = self.model.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
                pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = torch_functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)

    def warm_up(self, query: str, passage: str, rounds: int) -> None:
        for _ in range(rounds):
            self._encode([query], kind="query", batch_size=1)
            self._encode([passage], kind="passage", batch_size=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def encode_corpus(self, texts: list[str], batch_size: int) -> tuple[np.ndarray, float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        embeddings = self._encode(
            texts, kind="passage", batch_size=batch_size, show_progress=True
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return embeddings, elapsed_ms

    def encode_queries(self, texts: list[str]) -> tuple[np.ndarray, dict[str, float]]:
        vectors: list[np.ndarray] = []
        latencies: list[float] = []
        for text in texts:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            vectors.append(self._encode([text], kind="query", batch_size=1)[0])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter_ns() - started) / 1e6)
        return np.vstack(vectors), latency_percentiles(latencies)

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_model_artifact(path: Path, repository: str) -> None:
    """Verify the pinned acquisition manifest before executing local model code."""
    manifest_path = path / ".hh_goa_model.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Model artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != repository or not manifest.get("revision"):
        raise RuntimeError("Model artifact repository/revision metadata is invalid")
    python_files = {
        str(candidate.relative_to(path)).replace("\\", "/"): _sha256_file(candidate)
        for candidate in path.rglob("*.py")
        if candidate.is_file()
    }
    expected = manifest.get("python_file_sha256")
    if python_files and not isinstance(expected, dict):
        raise RuntimeError("Model Python-code hashes are missing; refusing remote-code execution")
    if python_files != (expected or {}):
        raise RuntimeError("Model Python-code hash verification failed")
