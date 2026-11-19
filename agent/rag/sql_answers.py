from __future__ import annotations

from decimal import Decimal
from typing import Any

from agent.services.database import db


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)

    return value


def _fetch_one(
    query: str,
    params: tuple[Any, ...],
) -> dict[str, Any] | None:
    """Execute one predefined read-only SQL query."""
    return db.fetch_one(query, params)


def _fetch_all(
    query: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Execute a predefined read-only SQL query."""
    return db.fetch_all(query, params)


def invoice_details(
    invoice_id: str,
) -> dict[str, Any] | None:
    """
    Retrieve persisted audit information for one invoice.
    """

    return _fetch_one(
        """
        SELECT
            invoice_id,
            invoice_number,
            vendor_name,
            po_number,
            invoice_date,
            currency,
            subtotal_amount,
            tax_amount,
            tax_rate,
            total_amount,
            validation_status,
            recommendation
        FROM audit.invoices
        WHERE invoice_id = %s
        LIMIT 1
        """,
        (invoice_id,),
    )


def invoice_discrepancies(
    invoice_id: str,
) -> list[dict[str, Any]]:
    """
    Return discrepancies recorded for an invoice.
    """

    return _fetch_all(
        """
        SELECT
            discrepancy_type,
            severity,
            message,
            details,
            status
        FROM audit.discrepancies
        WHERE invoice_id = %s AND UPPER(status) = 'OPEN'
        ORDER BY id
        """,
        (invoice_id,),
    )


def invoice_line_items(
    invoice_id: str,
) -> list[dict[str, Any]]:
    """
    Return effective persisted invoice line items.
    """

    return _fetch_all(
        """
        SELECT
            line_number,
            sku,
            description,
            quantity,
            unit_price,
            line_total,
            source
        FROM audit.invoice_line_items
        WHERE invoice_id = %s
        ORDER BY line_number
        """,
        (invoice_id,),
    )


def invoice_summary(
    invoice_id: str,
) -> dict[str, Any]:
    """
    Build a controlled structured response for an invoice.
    """

    invoice = invoice_details(invoice_id)

    if invoice is None:
        return {
            "found": False,
            "invoice_id": invoice_id,
            "message": "Invoice not found.",
        }

    discrepancies = invoice_discrepancies(
        invoice_id
    )

    line_items = invoice_line_items(
        invoice_id
    )

    return {
        "found": True,
        "invoice": invoice,
        "discrepancies": discrepancies,
        "line_items": line_items,
    }


def list_invoices(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Return recent invoices.

    The limit is clamped to prevent uncontrolled queries.
    """

    limit = max(
        1,
        min(int(limit), 100),
    )

    return _fetch_all(
        f"""
        SELECT
            invoice_id,
            invoice_number,
            vendor_name,
            po_number,
            invoice_date,
            currency,
            total_amount,
            validation_status,
            recommendation
        FROM audit.invoices
        ORDER BY created_at DESC
        LIMIT {limit}
        """
    )


def answer_structured_question(
    question: str,
    invoice_id: str | None = None,
) -> dict[str, Any]:
    """
    Answer supported structured questions using predefined
    SQL operations.

    No user-provided SQL is ever executed.
    """

    question_lower = question.lower().strip()

    if invoice_id:
        summary = invoice_summary(invoice_id)

        if not summary["found"]:
            return {
                "answer": summary["message"],
                "data": summary,
            }

        invoice = summary["invoice"]

        if (
            "discrepanc" in question_lower
            or "mismatch" in question_lower
            or "difference" in question_lower
        ):
            discrepancies = summary[
                "discrepancies"
            ]

            if not discrepancies:
                answer = (
                    f"Invoice {invoice_id} "
                    "has no recorded discrepancies."
                )
            else:
                answer = (
                    f"Invoice {invoice_id} has "
                    f"{len(discrepancies)} "
                    "recorded discrepancies."
                )

            return {
                "answer": answer,
                "data": discrepancies,
            }

        if (
            "line" in question_lower
            or "sku" in question_lower
            or "item" in question_lower
        ):
            return {
                "answer": (
                    f"Invoice {invoice_id} contains "
                    f"{len(summary['line_items'])} "
                    "line items."
                ),
                "data": summary["line_items"],
            }

        if (
            "total" in question_lower
            or "amount" in question_lower
            or "value" in question_lower
        ):
            total = invoice.get(
                "total_amount"
            )

            currency = invoice.get(
                "currency"
            ) or ""

            return {
                "answer": (
                    f"Invoice {invoice_id} total is "
                    f"{currency} {total}."
                ),
                "data": {
                    "invoice_id": invoice_id,
                    "total_amount": total,
                    "currency": currency,
                },
            }

        if (
            "status" in question_lower
            or "state" in question_lower
        ):
            status = invoice.get(
                "validation_status"
            )

            return {
                "answer": (
                    f"Invoice {invoice_id} "
                    f"validation status is "
                    f"{status}."
                ),
                "data": {
                    "invoice_id": invoice_id,
                    "validation_status": status,
                },
            }

        if (
            "recommendation" in question_lower
            or "decision" in question_lower
        ):
            recommendation = invoice.get(
                "recommendation"
            )

            return {
                "answer": (
                    f"Invoice {invoice_id} "
                    f"recommendation is "
                    f"{recommendation}."
                ),
                "data": {
                    "invoice_id": invoice_id,
                    "recommendation": recommendation,
                },
            }

        if (
            "vendor" in question_lower
            or "supplier" in question_lower
        ):
            vendor = invoice.get(
                "vendor_name"
            )

            return {
                "answer": (
                    f"Invoice {invoice_id} "
                    f"vendor is {vendor}."
                ),
                "data": {
                    "invoice_id": invoice_id,
                    "vendor_name": vendor,
                },
            }

        if (
            "purchase order" in question_lower
            or "po" in question_lower
        ):
            po_number = invoice.get(
                "po_number"
            )

            return {
                "answer": (
                    f"Invoice {invoice_id} "
                    f"references PO {po_number}."
                ),
                "data": {
                    "invoice_id": invoice_id,
                    "po_number": po_number,
                },
            }

        return {
            "answer": (
                f"Invoice {invoice_id} was found."
            ),
            "data": summary,
        }

    # No specific invoice was supplied.
    invoices = list_invoices()

    if not invoices:
        return {
            "answer": "No invoices were found.",
            "data": [],
        }

    return {
        "answer": (
            f"Found {len(invoices)} recent invoices."
        ),
        "data": invoices,
    }


def sql_answer_state(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    SQL branch adapter for the LangGraph state.
    """

    question = str(
        state.get("rag_question")
        or ""
    ).strip()

    filters = (
        state.get("rag_filters")
        or {}
    )

    invoice_id = filters.get(
        "invoice_id"
    )

    try:
        result = answer_structured_question(
            question=question,
            invoice_id=(
                str(invoice_id)
                if invoice_id
                else None
            ),
        )

        return {
            "rag_answer": result["answer"],
            "rag_sql_result": result["data"],
            "rag_answer_accepted": True,
            "rag_score": 1.0,
        }

    except Exception as exc:
        return {
            "rag_answer": (
                "Unable to answer the structured "
                "question."
            ),
            "rag_sql_result": {},
            "rag_answer_accepted": False,
            "warnings": [
                *list(
                    state.get("warnings") or []
                ),
                f"Structured SQL query failed: {exc}",
            ],
        }