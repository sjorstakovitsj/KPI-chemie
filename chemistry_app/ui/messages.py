"""Translate application exceptions into page-level messages."""
from __future__ import annotations
from collections.abc import Callable
from typing import TypeVar
import streamlit as st
from chemistry_app.infrastructure.exceptions import ChemistryAppError,DataArtifactNotFoundError,ReferenceTableValidationError
T=TypeVar("T")
def run_with_ui_errors(action: Callable[[],T]) -> T | None:
    try: return action()
    except DataArtifactNotFoundError as exc: st.error(f"Data-artifact ontbreekt: {exc}")
    except ReferenceTableValidationError as exc: st.error(f"Referentietabel is ongeldig: {exc}")
    except ChemistryAppError as exc: st.error(str(exc))
    return None
