
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.state import InvoiceState
from agent.nodes.monitor import monitor_invoice
from agent.nodes.extraction import extract_invoice
from agent.nodes.translation import translate_invoice
from agent.nodes.field_extraction import extract_fields
from agent.nodes.data_validation import data_validate
from agent.nodes.business_validation import business_validate
from agent.nodes.reporting import generate_report
from agent.nodes.persistence import persist_invoice

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from agent.config import settings
from agent.services.database import update_processing_stage


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

_checkpointer: Any = None
_checkpointer_connection: Any = None


def build_checkpointer() -> Any:
    """
    Build a PostgreSQL checkpointer so interrupted invoice workflows
    can be resumed after an API request or process restart.

    Use the same configured PostgreSQL connection as the audit database.
    """
    global _checkpointer, _checkpointer_connection

    # Use libpq keyword parameters instead of interpolating credentials into a URL.
    # This safely supports passwords containing @, :, /, or other URL-reserved characters.
    _checkpointer_connection = psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        autocommit=True,
    )

    _checkpointer = PostgresSaver(_checkpointer_connection)
    _checkpointer.setup()

    return _checkpointer


# ---------------------------------------------------------------------
# Workflow routing
# ---------------------------------------------------------------------

def route_after_monitor(state: InvoiceState) -> str:
    if state.get("is_duplicate"):
        return "end"

    if state.get("processing_status") in {"idle", "failed"}:
        return "end"

    return "extract"


def route_after_extraction(state: InvoiceState) -> str:
    if state.get("processing_status") == "failed":
        return "end"

    return "translate"


def route_after_translation(state: InvoiceState) -> str:
    if state.get("processing_status") == "failed":
        return "end"

    return "field_extraction"


def route_after_data_validation(state: InvoiceState) -> str:
    """
    Business validation is performed after data validation, even when
    data validation finds issues. Both sets of results are useful to
    the human reviewer.
    """
    if state.get("processing_status") == "failed":
        return "end"

    return "business_validation"

def route_after_business_validation(state: InvoiceState) -> str:
    if state.get("processing_status") == "failed":
        return "end"
    # Missing ERP master data is a hard business stop: report the invalid
    # state, persist it, and do not continue to human correction/RAG.
    if state.get("missing_erp_records") or state.get("erp_lookup_failed"):
        return "reporting"
    return "reporting"

# def route_after_business_validation(state: InvoiceState) -> str:
#     if state.get("processing_status") == "failed":
#         return "end"

#     if (
#         state.get("human_review_required")
#         or not state.get("data_validation_passed", False)
#         or not state.get("business_validation_passed", False)
#         or state.get("missing_erp_records")
#         or state.get("erp_lookup_failed")
#     ):
#         return "human_review"

#     return "reporting"


def route_after_human_review(state: InvoiceState) -> str:
    """
    After resume, the human-review node marks the state for revalidation.
    A rejected invoice goes to reporting; otherwise rerun data validation
    and business validation using corrected values.
    """
    decision = str(state.get("human_decision") or "").strip().lower()

    if decision in {"reject", "rejected"}:
        return "reporting"

    if state.get("human_review_action") == "revalidate":
        return "data_validation"

    return "reporting"


def route_after_reporting(state: InvoiceState) -> str:
    if state.get("processing_status") == "failed":
        return "end"

    return "persistence"

def route_after_persistence(state: InvoiceState) -> str:
    """Persist first, then pause for HITL, otherwise index the current truth."""
    if state.get("persistence_status") != "persisted":
        return "end"
    if state.get("missing_erp_records") or state.get("erp_lookup_failed"):
        return "end"
    if state.get("human_review_required") and not state.get("human_decision"):
        return "human_review"
    return "rag_indexing"

# def route_after_persistence(state: InvoiceState) -> str:
#     if state.get("persistence_status") != "persisted":
#         return "end"

#     return "rag_indexing"


# ---------------------------------------------------------------------
# Human-review pause
# ---------------------------------------------------------------------

def human_review_pause(state: InvoiceState) -> dict[str, Any]:
    """
    Pause execution and save a checkpoint until a human submits a
    decision, feedback, or corrections through the API.

    The API must resume this graph using Command(resume=payload) and
    the same configurable thread_id used to start the invoice.
    """
    review_payload = {
        "invoice_id": state.get("invoice_id"),
        "validation_status": state.get("validation_status"),
        "recommendation": state.get("recommendation"),
        "data_validation_passed": state.get("data_validation_passed"),
        "business_validation_passed": state.get(
            "business_validation_passed"
        ),
        "data_validation_errors": state.get("data_validation_errors", []),
        "business_validation_errors": state.get(
            "business_validation_errors", []
        ),
        "missing_fields": state.get("missing_fields", []),
        "missing_erp_records": state.get("missing_erp_records", []),
        "discrepancies": state.get("discrepancies", []),
        "human_feedback": state.get("human_feedback", []),
        "human_corrections": state.get("human_corrections", {}),
    }

    human_response = interrupt(review_payload)

    if not isinstance(human_response, dict):
        return {
            "processing_status": "human_review_invalid_response",
            "errors": [
                "Human review response must be a JSON object."
            ],
            "human_review_required": True,
        }

    corrections = human_response.get("corrections") or {}
    feedback = human_response.get("feedback")
    decision = human_response.get("decision")

    if not isinstance(corrections, dict):
        return {
            "processing_status": "human_review_invalid_response",
            "errors": ["Human corrections must be a JSON object."],
            "human_review_required": True,
        }

    existing_corrections = dict(state.get("human_corrections") or {})
    existing_corrections.update(corrections)

    existing_feedback = state.get("human_feedback") or []

    if isinstance(existing_feedback, dict):
        existing_feedback = [existing_feedback]
    else:
        existing_feedback = list(existing_feedback)

    if feedback:
        existing_feedback.append(
            {
                "feedback": str(feedback),
                "decision": decision,
            }
        )

    normalized_decision = (
        str(decision).strip().lower()
        if decision is not None
        else ""
    )

    if normalized_decision in {"reject", "rejected"}:
        return {
            "human_decision": "rejected",
            "human_feedback": existing_feedback,
            "human_corrections": existing_corrections,
            "human_review_required": False,
            "human_review_action": "finish",
            "processing_status": "human_review_rejected",
        }

    if normalized_decision in {"approve", "approved"} and not corrections:
        # Approval must not bypass failed checks. Route back through
        # validation and only approve if all checks now pass.
        return {
            "human_decision": "approved",
            "human_feedback": existing_feedback,
            "human_corrections": existing_corrections,
            "human_review_required": True,
            "human_review_action": "revalidate",
            "processing_status": "human_review_submitted",
        }

    if normalized_decision not in {
        "",
        "correct",
        "corrected",
        "revalidate",
        "approve",
        "approved",
    }:
        return {
            "human_decision": normalized_decision,
            "human_feedback": existing_feedback,
            "human_corrections": existing_corrections,
            "human_review_required": True,
            "human_review_action": "finish",
            "processing_status": "human_review_invalid_decision",
            "errors": [
                f"Unsupported human-review decision: {decision}"
            ],
        }

    return {
        "human_decision": normalized_decision or "corrected",
        "human_feedback": existing_feedback,
        "human_corrections": existing_corrections,
        "human_review_required": True,
        "human_review_action": "revalidate",
        "force_revalidation": True,
        "processing_status": "human_review_submitted",
    }


# ---------------------------------------------------------------------
# RAG indexing
# ---------------------------------------------------------------------

def rag_indexing_node(state: InvoiceState) -> dict[str, Any]:
    """
    Index the final invoice after persistence. RAG failures should not
    erase or invalidate the saved audit result.
    """
    try:
        from agent.rag.indexing import index_invoice

        result = index_invoice(state)

        if isinstance(result, dict):
            return {
                **result,
                "processing_status": "complete",
            }

        return {
            "processing_status": "complete",
        }

    except Exception as exc:
        warnings = list(state.get("warnings") or [])
        warnings.append(f"RAG indexing failed: {exc}")

        return {
            "warnings": warnings,
            "processing_status": "complete",
        }


# ---------------------------------------------------------------------
# Live stage tracking
# ---------------------------------------------------------------------

_STAGE_META = {
    "monitor": ("monitoring", "Monitoring / queued", 10),
    "extract": ("extraction", "Extraction / OCR", 25),
    "translate": ("translation", "Translation", 40),
    "field_extraction": ("field_extraction", "Field extraction", 50),
    "data_validation": ("validation", "Invoice validation", 65),
    "business_validation": ("business_validation", "ERP business validation", 78),
    "reporting": ("reporting", "Report generation", 88),
    "persistence": ("persistence", "Saving audit record", 94),
    "human_review": ("human_review", "Human review required", 96),
    "rag_indexing": ("rag", "RAG indexing", 100),
}

def _tracked(name: str, fn):
    stage, label, progress = _STAGE_META[name]
    def wrapped(state):
        update_processing_stage(state.get("invoice_id"), state.get("file_name"), state.get("file_checksum"),
                               stage, label, progress, "processing", f"Running {label.lower()}.")
        result = fn(state)
        if isinstance(result, dict):
            status = str(result.get("processing_status") or "processing")
            error = None if status not in {"failed", "human_review_invalid_response"} else "; ".join(map(str, result.get("errors") or []))
            result_status = str(result.get("processing_status") or "")
            if stage == "human_review" and not result.get("human_review_action"):
                live_status = "waiting"
            elif error:
                live_status = "failed"
            elif result_status in {"complete", "persistence_complete", "duplicate"} or result.get("rag_indexed"):
                live_status = "complete"
            else:
                live_status = "processing"
            update_processing_stage(result.get("invoice_id") or state.get("invoice_id"), state.get("file_name"), state.get("file_checksum"),
                                   stage, label, progress, live_status,
                                   str(result.get("message") or f"{label} completed."), error)
        return result
    return wrapped

# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def build_invoice_graph():
    builder = StateGraph(InvoiceState)

    builder.add_node("monitor", _tracked("monitor", monitor_invoice))
    builder.add_node("extract", _tracked("extract", extract_invoice))
    builder.add_node("translate", _tracked("translate", translate_invoice))
    builder.add_node("field_extraction", _tracked("field_extraction", extract_fields))
    builder.add_node("data_validation", _tracked("data_validation", data_validate))
    builder.add_node("business_validation", _tracked("business_validation", business_validate))
    builder.add_node("human_review", _tracked("human_review", human_review_pause))
    builder.add_node("reporting", _tracked("reporting", generate_report))
    builder.add_node("persistence", _tracked("persistence", persist_invoice))
    builder.add_node("rag_indexing", _tracked("rag_indexing", rag_indexing_node))

    builder.add_edge(START, "monitor")

    builder.add_conditional_edges(
        "monitor",
        route_after_monitor,
        {
            "end": END,
            "extract": "extract",
        },
    )

    builder.add_conditional_edges(
        "extract",
        route_after_extraction,
        {
            "end": END,
            "translate": "translate",
        },
    )

    builder.add_conditional_edges(
        "translate",
        route_after_translation,
        {
            "end": END,
            "field_extraction": "field_extraction",
        },
    )

    builder.add_edge("field_extraction", "data_validation")

    builder.add_conditional_edges(
        "data_validation",
        route_after_data_validation,
        {
            "end": END,
            "business_validation": "business_validation",
        },
    )

    builder.add_conditional_edges(
        "business_validation",
        route_after_business_validation,
        {
            "end": END,
            "reporting": "reporting",
        },
    )

    # builder.add_conditional_edges(
    #     "business_validation",
    #     route_after_business_validation,
    #     {
    #         "end": END,
    #         "human_review": "human_review",
    #         "reporting": "reporting",
    #     },
    # )

    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "data_validation": "data_validation",
            "reporting": "reporting",
        },
    )

    builder.add_conditional_edges(
        "reporting",
        route_after_reporting,
        {
            "end": END,
            "persistence": "persistence",
        },
    )

    builder.add_conditional_edges(
        "persistence",
        route_after_persistence,
        {
            "end": END,
            "human_review": "human_review",
            "rag_indexing": "rag_indexing",
        },
    )
    # builder.add_conditional_edges(
    #     "persistence",
    #     route_after_persistence,
    #     {
    #         "end": END,
    #         "rag_indexing": "rag_indexing",
    #     },
    # )

    builder.add_edge("rag_indexing", END)

    return builder.compile(checkpointer=build_checkpointer())


invoice_graph = build_invoice_graph()


# ---------------------------------------------------------------------
# Public invocation helpers
# ---------------------------------------------------------------------

def _thread_config(state: dict[str, Any]) -> dict[str, Any]:
    """
    Use a stable thread identifier so a pending invoice can be resumed.
    """
    thread_id = (
        state.get("thread_id")
        or state.get("invoice_id")
    )

    if not thread_id:
        raise ValueError(
            "thread_id or invoice_id is required for graph execution."
        )

    return {
        "configurable": {
            "thread_id": str(thread_id),
        }
    }


def process_invoice(state: dict[str, Any]) -> dict[str, Any]:
    """
    Start or invoke the invoice workflow.

    To resume an interrupted workflow, use resume_invoice() with the
    same invoice_id/thread_id rather than calling this function again.
    """
    return invoice_graph.invoke(
        state,
        config=_thread_config(state),
    )


def resume_invoice(
    invoice_id: str,
    response: dict[str, Any],
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """
    Resume a checkpointed invoice after human review.

    `response` is passed to the pending interrupt. The caller must first
    verify that the requester is authorized to submit this decision.
    """
    if not isinstance(response, dict):
        raise ValueError("Human review response must be a JSON object.")

    resolved_thread_id = thread_id or invoice_id

    return invoice_graph.invoke(
        Command(resume=response),
        config={
            "configurable": {
                "thread_id": str(resolved_thread_id),
            }
        },
    )
