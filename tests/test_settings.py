"""
tests/test_settings.py
----------------------
Validates that settings.py has the correct default values.
These tests run without any .env file — they verify the hardcoded defaults.
"""

import os
import pytest


def test_primary_llm_model_name():
    """PRIMARY_LLM must default to the exact Groq model ID string."""
    # Unset env var to test the default, not whatever is in .env
    os.environ.pop("PRIMARY_LLM", None)

    # Re-import to pick up the cleared env var
    import importlib
    import config.settings as settings
    importlib.reload(settings)

    assert settings.PRIMARY_LLM == "llama-3.3-70b-versatile", (
        f"Expected 'llama-3.3-70b-versatile', got '{settings.PRIMARY_LLM}'. "
        "If Groq renamed this model, update the default in settings.py."
    )


def test_fallback_llm_model_name():
    """FALLBACK_LLM must default to the exact Groq fallback model ID string."""
    os.environ.pop("FALLBACK_LLM", None)

    import importlib
    import config.settings as settings
    importlib.reload(settings)

    assert settings.FALLBACK_LLM == "llama-3.1-8b-instant"


def test_contradiction_threshold_default():
    """Default contradiction threshold must be 0.75."""
    os.environ.pop("CONTRADICTION_THRESHOLD", None)

    import importlib
    import config.settings as settings
    importlib.reload(settings)

    assert settings.CONTRADICTION_THRESHOLD == 0.75


def test_max_nli_pairs_default():
    """MAX_NLI_PAIRS must default to 50 to stay within the 15s query SLA."""
    os.environ.pop("MAX_NLI_PAIRS", None)

    import importlib
    import config.settings as settings
    importlib.reload(settings)

    assert settings.MAX_NLI_PAIRS == 50


def test_vector_store_default_is_chromadb():
    """Default vector store backend must be chromadb for local dev."""
    os.environ.pop("VECTOR_STORE_BACKEND", None)

    import importlib
    import config.settings as settings
    importlib.reload(settings)

    assert settings.VECTOR_STORE_BACKEND == "chromadb"


def test_db_backend_default_is_sqlite():
    """Default DB backend must be sqlite for local dev."""
    os.environ.pop("DB_BACKEND", None)

    import importlib
    import config.settings as settings
    importlib.reload(settings)

    assert settings.DB_BACKEND == "sqlite"
