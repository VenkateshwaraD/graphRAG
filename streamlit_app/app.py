import os

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}

st.set_page_config(page_title="University GraphRAG Assistant", layout="wide")
st.title("University Knowledge Assistant (GraphRAG)")

page = st.sidebar.radio("Section", ["Chat", "Documents", "Graph Index", "Review"])


def _path_dot(hop_path, edge_directions):
    """Build a graphviz DOT for one cited hop path."""
    lines = ["digraph {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    dirs = edge_directions or []
    for name in hop_path:
        lines.append(f'  "{name}";')
    for idx in range(len(hop_path) - 1):
        src, tgt = hop_path[idx], hop_path[idx + 1]
        label = dirs[idx] if idx < len(dirs) else ""
        lines.append(f'  "{src}" -> "{tgt}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def _subgraph_dot(sub):
    """Build a graphviz DOT for a /graph/subgraph response."""
    lines = ["digraph {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    for node in sub.get("nodes", []):
        lines.append(f'  "{node["name"]}";')
    for edge in sub.get("edges", []):
        lines.append(
            f'  "{edge["source"]}" -> "{edge["target"]}" [label="{edge.get("description") or ""}"];'
        )
    lines.append("}")
    return "\n".join(lines)

if page == "Documents":
    st.header("Upload documents")
    uploaded = st.file_uploader("pdf / docx / txt / md", type=["pdf", "docx", "txt", "md"])
    title = st.text_input("Title (optional)")
    classification = st.selectbox(
        "Classification",
        ["internal", "public", "restricted"],
        help="public = all tenant users · internal = non-guests · "
        "restricted = owner/admins + explicit grants only",
    )
    dept_opts = {
        dept["name"]: dept["id"]
        for dept in requests.get(f"{API}/departments", headers=HEADERS, timeout=30).json()
    }
    course_opts = {
        crs["code"]: crs["id"]
        for crs in requests.get(f"{API}/courses", headers=HEADERS, timeout=30).json()
    }
    dept = st.selectbox("Department scope (optional)", [""] + list(dept_opts))
    course = st.selectbox("Course scope (optional)", [""] + list(course_opts))
    if uploaded and st.button("Upload & chunk"):
        params = {"title": title or uploaded.name, "classification": classification}
        if dept:
            params["department_id"] = dept_opts[dept]
        if course:
            params["course_id"] = course_opts[course]
        with st.spinner("Extracting, chunking, embedding…"):
            result = requests.post(
                f"{API}/documents",
                files={"file": (uploaded.name, uploaded.getvalue())},
                params=params,
                headers=HEADERS,
                timeout=300,
            )
        if result.ok:
            st.success(
                f"Uploaded {result.json()['title']} → status {result.json()['status']}"
            )
        else:
            st.error(result.text)

    st.subheader("Indexed documents")
    docs = requests.get(f"{API}/documents", headers=HEADERS, timeout=30).json()
    for doc in docs:
        st.write(
            f"- **{doc['title']}** — `{doc['status']}` ({doc['filename']})"
            f" — `{doc.get('classification', 'internal')}`"
        )

elif page == "Graph Index":
    st.header("Knowledge graph")
    stats = requests.get(f"{API}/graph/stats", headers=HEADERS, timeout=30).json()
    cols = st.columns(6)
    for col, key in zip(
        cols,
        ["entities", "relationships", "communities", "text_units", "community_reports"], strict=False,
    ):
        col.metric(key.replace("_", " "), stats.get(key, 0))
    st.write(f"Indexed: **{stats.get('indexed')}**")
    if stats.get("stale"):
        st.warning(
            f"Graph index is stale — {stats.get('stale_documents')} deleted "
            "document(s) remain in the artifacts until a rebuild. Their "
            "content is already unreachable; run the index to clean up."
        )

    if st.button("Run GraphRAG index"):
        result = requests.post(f"{API}/ingest", headers=HEADERS, timeout=30)
        if result.ok:
            st.info(f"Job {result.json()['id']} started — refresh to see progress")
        else:
            st.error(result.text)

    jobs = requests.get(f"{API}/ingest", headers=HEADERS, timeout=30).json()
    if jobs:
        st.subheader("Jobs")
        for job in jobs[:5]:
            st.write(f"- `{job['status']}` {job['created_at']} {job.get('error') or ''}")
        if jobs[0].get("stats"):
            st.json(jobs[0]["stats"])

    ents = requests.get(f"{API}/graph/entities", params={"limit": 50}, headers=HEADERS, timeout=30).json()
    if ents:
        st.subheader("Top entities")
        st.dataframe(ents, use_container_width=True)

        names = [ent["name"] for ent in ents]
        focus = st.selectbox("Explore neighborhood", names)
        depth = st.slider("depth", 1, 3, 1)
        sub = requests.get(
            f"{API}/graph/subgraph/{focus}",
            params={"depth": depth},
            headers=HEADERS,
            timeout=30,
        )
        if sub.ok:
            st.graphviz_chart(_subgraph_dot(sub.json()))
        else:
            st.error(sub.text)

elif page == "Review":
    st.header("Relationship review")
    st.caption(
        "Approve or reject extracted relationships. Rejected edges are "
        "excluded from traversal immediately — no reindex needed."
    )
    queue = requests.get(
        f"{API}/graph/review/queue", params={"limit": 25}, headers=HEADERS, timeout=30
    ).json()
    if not queue:
        st.info("Nothing left to review.")
    for item in queue:
        with st.container(border=True):
            st.write(
                f"**{item['source']}** → **{item['target']}**  "
                f"(weight {item.get('weight')})"
            )
            st.write(item.get("description") or "")
            cols = st.columns(3)
            if cols[0].button("Approve", key=f"ok_{item['rel_hash']}"):
                requests.post(
                    f"{API}/graph/review",
                    json={
                        "source": item["source"],
                        "target": item["target"],
                        "description": item.get("description") or "",
                        "verdict": "approved",
                    },
                    headers=HEADERS,
                    timeout=30,
                )
                st.rerun()
            if cols[1].button("Reject", key=f"no_{item['rel_hash']}"):
                requests.post(
                    f"{API}/graph/review",
                    json={
                        "source": item["source"],
                        "target": item["target"],
                        "description": item.get("description") or "",
                        "verdict": "rejected",
                    },
                    headers=HEADERS,
                    timeout=30,
                )
                st.rerun()

else:
    mode = st.sidebar.selectbox(
        "Retrieval mode (auto = router decides)",
        ["auto", "hybrid", "traverse", "local", "global"],
    )
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.convo = None

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("route"):
                cap = f"route: {msg['route']}"
                if msg.get("requested_route") and msg["requested_route"] != msg["route"]:
                    cap += f" (requested: {msg['requested_route']})"
                if msg.get("route_reason"):
                    cap += f" — {msg['route_reason']}"
                st.caption(cap)
            if msg.get("warning"):
                st.warning(msg["warning"])
            for citation in msg.get("citations", []):
                label = f"[{citation['n']}] {citation.get('document_title') or citation.get('document_id')}"
                if citation.get("score") is not None:
                    label += f"  (score {citation['score']:.2f})"
                if citation.get("supported") is False:
                    label += " (unsupported)"
                with st.expander(label):
                    st.write(citation["excerpt"])
                    if citation.get("hop_path"):
                        st.graphviz_chart(_path_dot(citation["hop_path"], citation.get("edge_directions")))
                    for alt in citation.get("alt_paths") or []:
                        st.caption("alt path: " + " → ".join(alt))
                    if citation.get("years"):
                        st.caption("years: " + ", ".join(map(str, citation["years"])))

    if q := st.chat_input("Ask a question (try a multi-hop one)"):
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Routing & retrieving…"):
                result = requests.post(
                    f"{API}/chat",
                    json={
                        "question": q,
                        "conversation_id": st.session_state.convo,
                        "mode": None if mode == "auto" else mode,
                    },
                    headers=HEADERS,
                    timeout=600,
                )
            if result.ok:
                data = result.json()
                st.session_state.convo = data["conversation_id"]
                st.markdown(data["answer"])
                cap = f"route: {data['route']}"
                if data.get("requested_route") and data["requested_route"] != data["route"]:
                    cap += f" (requested: {data['requested_route']})"
                if data.get("route_reason"):
                    cap += f" — {data['route_reason']}"
                st.caption(cap)
                if data.get("warning"):
                    st.warning(data["warning"])
                for citation in data["citations"]:
                    label = f"[{citation['n']}] {citation.get('document_title') or 'source'}"
                    if citation.get("score") is not None:
                        label += f"  (score {citation['score']:.2f})"
                    if citation.get("supported") is False:
                        label += " (unsupported)"
                    with st.expander(label):
                        st.write(citation["excerpt"])
                        if citation.get("hop_path"):
                            st.graphviz_chart(
                                _path_dot(
                                    citation["hop_path"],
                                    citation.get("edge_directions"),
                                )
                            )
                        for alt in citation.get("alt_paths") or []:
                            st.caption("alt path: " + " → ".join(alt))
                        if citation.get("years"):
                            st.caption(
                                "years: " + ", ".join(map(str, citation["years"]))
                            )
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "route": data["route"],
                        "requested_route": data.get("requested_route"),
                        "route_reason": data.get("route_reason"),
                        "warning": data.get("warning"),
                        "citations": data["citations"],
                    }
                )
            else:
                st.error(result.text)
