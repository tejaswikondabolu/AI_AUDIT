from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import requests


AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8000").rstrip("/")
ERP_URL = os.getenv("ERP_URL", "http://mock-erp:8000").rstrip("/")


class AgentAPIError(Exception):
    pass


def _request(
    method: str,
    path: str,
    **kwargs: Any,
) -> Any:

    url = f"{AGENT_URL}{path}"

    try:
        default_timeout = 180 if path == "/api/query" or path.endswith("/revalidate") else 60 if path == "/api/invoices/upload" else 15
        timeout = kwargs.pop("timeout", default_timeout)
        response = requests.request(
            method,
            url,
            timeout=timeout,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise AgentAPIError(
            f"Unable to connect to Agent API: {exc}"
        ) from exc

    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text

        raise AgentAPIError(
            f"Agent API returned {response.status_code}: {detail}"
        )

    if not response.content:
        return {}

    try:
        return response.json()
    except Exception:
        return response.text


def health() -> dict[str, Any]:
    return _request(
        "GET",
        "/health",
    )


def get_dashboard_summary() -> dict[str, Any]:
    return _request(
        "GET",
        "/api/dashboard/summary",
    )


def get_invoices() -> list[dict[str, Any]]:
    return _request(
        "GET",
        "/api/invoices",
    )


def get_upload_status() -> dict[str, Any]:
    return _request(
        "GET",
        "/api/uploads/status",
    )


def get_invoice(
    invoice_id: str,
) -> dict[str, Any]:

    return _request(
        "GET",
        f"/api/invoices/{invoice_id}",
    )


def get_report(
    invoice_id: str,
) -> dict[str, Any]:

    return _request(
        "GET",
        f"/api/invoices/{invoice_id}/report",
    )


def upload_invoice(
    filename: str,
    content: bytes,
) -> dict[str, Any]:

    return _request(
        "POST",
        "/api/invoices/upload",
        files={
            "file": (
                filename,
                content,
                "application/octet-stream",
            )
        },
    )


def revalidate_invoice(
    invoice_id: str,
    corrections: dict[str, Any] | None = None,
    decision: str | None = None,
    feedback: str | None = None,
) -> dict[str, Any]:

    return _request(
        "POST",
        f"/api/invoices/{invoice_id}/revalidate",
        json={
            "corrections": corrections or {},
            "decision": decision,
            "feedback": feedback,
        },
    )



def clear_pending_queue() -> dict[str, Any]:
    return _request("DELETE", "/api/uploads/queue")


def reset_invoice_workspace(confirmation: str) -> dict[str, Any]:
    return _request("POST", "/api/admin/reset-invoice-workspace", json={"confirmation": confirmation}, timeout=60)


def get_erp_purchase_order(po_number: str) -> dict[str, Any]:
    """Read the reference PO from the mock ERP service, without modifying it."""
    url = f"{ERP_URL}/po/{quote(po_number, safe='')}"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        raise AgentAPIError(f"Unable to connect to ERP reference data: {exc}") from exc
    if response.status_code == 404:
        raise AgentAPIError(f"Purchase order {po_number} was not found in ERP.")
    if not response.ok:
        raise AgentAPIError(f"ERP returned {response.status_code}: {response.text[:500]}")
    try:
        return response.json()
    except ValueError as exc:
        raise AgentAPIError("ERP returned an unreadable purchase-order response.") from exc


def save_human_feedback(invoice_id: str, feedback: str) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/invoices/{invoice_id}/feedback",
        json={"feedback": feedback},
    )


def get_comparison(invoice_id: str) -> dict[str, Any]:
    return _request("GET", f"/api/invoices/{invoice_id}/comparison")


def ask_ai(question: str, invoice_id: str | None = None) -> dict[str, Any]:
    """Ask a question using the agent's SQL/RAG query endpoint."""
    return _request(
        "POST",
        "/api/query",
        json={"question": question, "invoice_id": invoice_id},
    )


def get_rag_evaluations(invoice_id: str) -> list[dict[str, Any]]:
    return _request("GET", f"/api/invoices/{invoice_id}/rag-evaluations")
