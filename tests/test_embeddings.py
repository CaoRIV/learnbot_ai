import sys
from types import SimpleNamespace

import core.embeddings as embeddings


EXPECTED_MULTILINGUAL_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def test_default_embedding_model_supports_multilingual_semantic_search():
    assert embeddings.EMBED_MODEL_NAME == EXPECTED_MULTILINGUAL_MODEL


def test_get_embed_model_loads_the_configured_default(monkeypatch):
    loaded_models = []

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            loaded_models.append(model_name)

        @staticmethod
        def get_sentence_embedding_dimension():
            return 384

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    embeddings.get_embed_model.cache_clear()

    try:
        model = embeddings.get_embed_model()
    finally:
        embeddings.get_embed_model.cache_clear()

    assert isinstance(model, FakeSentenceTransformer)
    assert loaded_models == [EXPECTED_MULTILINGUAL_MODEL]


def test_ui_reports_the_active_embedding_model():
    from rag_demo import get_system_models_info

    assert get_system_models_info()["Mô hình embedding"] == embeddings.EMBED_MODEL_NAME
