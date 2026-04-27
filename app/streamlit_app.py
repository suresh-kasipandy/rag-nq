"""Streamlit portfolio UI for the RAG NQ Showcase."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui.api_client import ApiClientError, RagApiClient, RetrievalMode
from app.ui.display import answer_display_text, hit_rank_summary, hit_title
from app.ui.eval_report import load_eval_report, rows_as_dicts, summarize_eval_report
from src.models.query_schemas import PassageHit, QueryResponse

MODES: tuple[RetrievalMode, RetrievalMode, RetrievalMode] = ("dense", "sparse", "hybrid")


def main() -> None:
    st.set_page_config(page_title="RAG NQ Showcase", layout="wide")
    st.title("RAG NQ Showcase")
    st.caption("Streamlit demo over the FastAPI surface and persisted eval artifacts.")

    api_base_url = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")
    eval_path = Path(st.sidebar.text_input("Eval report path", "artifacts/retrieval_eval.json"))
    client = RagApiClient(base_url=api_base_url)

    playground_tab, inspector_tab, eval_tab, system_tab = st.tabs(
        ["Query Playground", "Retrieval Inspector", "Eval Dashboard", "System / Runbook"]
    )

    with playground_tab:
        _render_query_playground(client)
    with inspector_tab:
        _render_retrieval_inspector(client)
    with eval_tab:
        _render_eval_dashboard(eval_path)
    with system_tab:
        _render_system_panel(client)


def _render_query_playground(client: RagApiClient) -> None:
    st.subheader("Query Playground")
    query = st.text_input("Question", "What is the capital of France?", key="playground_query")
    mode = st.selectbox("Retrieval mode", MODES, index=2, key="playground_mode")
    top_k = st.slider("Top K", min_value=1, max_value=50, value=10, key="playground_top_k")
    generate = st.checkbox("Generate grounded answer", value=True)
    if st.button("Run query", type="primary"):
        try:
            response = client.query(
                query=query,
                mode=mode,
                top_k=top_k,
                generate=generate,
            )
        except ApiClientError as exc:
            st.error(str(exc))
            return
        _render_query_response(response)


def _render_retrieval_inspector(client: RagApiClient) -> None:
    st.subheader("Retrieval Inspector")
    query = st.text_input("Question", "What is the capital of France?", key="inspect_query")
    top_k = st.slider("Top K", min_value=1, max_value=50, value=10, key="inspect_top_k")
    if st.button("Compare modes"):
        columns = st.columns(len(MODES))
        for column, mode in zip(columns, MODES, strict=True):
            with column:
                st.markdown(f"### {mode}")
                try:
                    response = client.retrieve(query=query, mode=mode, top_k=top_k)
                except ApiClientError as exc:
                    st.error(str(exc))
                    continue
                _render_hits(response.retrieved_passages or [])
                if response.retrieval_metrics is not None:
                    st.json(response.retrieval_metrics.model_dump())


def _render_eval_dashboard(eval_path: Path) -> None:
    st.subheader("Eval Dashboard")
    if not eval_path.is_file():
        st.warning(f"Eval report not found: {eval_path}")
        return
    try:
        summary = summarize_eval_report(load_eval_report(eval_path))
    except (OSError, ValueError) as exc:
        st.error(f"Could not load eval report: {exc}")
        return

    if summary.rows:
        st.dataframe(rows_as_dicts(summary.rows), use_container_width=True)
        cols = st.columns(3)
        for column, metric_name in zip(
            cols,
            ("recall_at_k", "mrr_at_k", "ndcg_at_k"),
            strict=True,
        ):
            winner = summary.winners[metric_name]
            column.metric(
                metric_name,
                f"{getattr(winner, metric_name):.3f}",
                f"{winner.mode} @ {winner.k}",
            )
    with st.expander("Run config", expanded=False):
        st.json(summary.run_config)


def _render_system_panel(client: RagApiClient) -> None:
    st.subheader("System / Runbook")
    try:
        health = client.health()
        config = client.config()
    except ApiClientError as exc:
        st.error(str(exc))
        return

    st.success(f"{health.service}: {health.status}")
    with st.expander("Safe runtime config", expanded=True):
        st.json(config.model_dump())

    st.markdown("### Local commands")
    eval_command = (
        "uv run python -m src.scripts.eval_retrieval "
        "--k-values 1,5,10,20 --modes dense,sparse,hybrid --max-queries 200"
    )
    st.code(
        "\n".join(
            [
                "docker compose up -d",
                "uv run python -m src.scripts.doctor",
                "uv run uvicorn app.api.main:app",
                eval_command,
            ]
        ),
        language="bash",
    )


def _render_query_response(response: QueryResponse) -> None:
    if response.grounded is not None:
        display_text = answer_display_text(response.grounded)
        if response.grounded.abstained:
            st.warning(display_text)
        else:
            st.markdown(display_text)
            st.caption(
                "Citations: "
                + ", ".join(citation.point_id for citation in response.grounded.citations)
            )
    _render_hits(response.retrieved_passages or [])
    if response.retrieval_metrics is not None:
        with st.expander("Retrieval metrics", expanded=False):
            st.json(response.retrieval_metrics.model_dump())


def _render_hits(hits: list[PassageHit]) -> None:
    if not hits:
        st.info("No retrieved passages.")
        return
    for hit in hits:
        with st.expander(hit_title(hit), expanded=False):
            st.caption(hit_rank_summary(hit))
            st.write(hit.text)
            if hit.context_text:
                st.markdown("**Context**")
                st.write(hit.context_text)
            st.json(
                {
                    "point_id": hit.point_id,
                    "group_id": hit.group_id,
                    "chunk_kind": hit.chunk_kind,
                    "source_row_ordinal": hit.source_row_ordinal,
                    "candidate_span": [hit.start_candidate_idx, hit.end_candidate_idx],
                    "duplicate_aliases": [
                        alias.model_dump() for alias in hit.duplicate_aliases
                    ],
                }
            )


if __name__ == "__main__":
    main()
