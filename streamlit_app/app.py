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
    for i in range(len(hop_path) - 1):
        u, v = hop_path[i], hop_path[i + 1]
        label = dirs[i] if i < len(dirs) else ""
        lines.append(f'  "{u}" -> "{v}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def _subgraph_dot(sub):
    """Build a graphviz DOT for a /graph/subgraph response."""
    lines = ["digraph {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    for n in sub.get("nodes", []):
        lines.append(f'  "{n["name"]}";')
    for e in sub.get("edges", []):
        lines.append(
            f'  "{e["source"]}" -> "{e["target"]}" [label="{e.get("description") or ""}"];'
        )
    lines.append("}")
    return "\n".join(lines)

if page == "Documents":
    st.header("Upload documents")
    uploaded = st.file_uploader("pdf / docx / txt / md", type=["pdf", "docx", "txt", "md"])
    title = st.text_input("Title (optional)")
    doc_group = st.text_input("Access group (optional)", value="default")
    if uploaded and st.button("Upload & chunk"):
        with st.spinner("Extracting, chunking, embedding…"):
            r = requests.post(
                f"{API}/documents",
                files={"file": (uploaded.name, uploaded.getvalue())},
                params={"title": title or uploaded.name, "doc_group": doc_group},
                headers=HEADERS,
                timeout=300,
            )
        if r.ok:
            st.success(f"Uploaded {r.json()['title']} → status {r.json()['status']}")
        else:
            st.error(r.text)

    st.subheader("Indexed documents")
    docs = requests.get(f"{API}/documents", headers=HEADERS, timeout=30).json()
    for d in docs:
        st.write(
            f"- **{d['title']}** — `{d['status']}` ({d['filename']})"
            f" — group `{d.get('doc_group', 'default')}`"
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

    if st.button("Run GraphRAG index"):
        r = requests.post(f"{API}/ingest", headers=HEADERS, timeout=30)
        if r.ok:
            st.info(f"Job {r.json()['id']} started — refresh to see progress")
        else:
            st.error(r.text)

    jobs = requests.get(f"{API}/ingest", headers=HEADERS, timeout=30).json()
    if jobs:
        st.subheader("Jobs")
        for j in jobs[:5]:
            st.write(f"- `{j['status']}` {j['created_at']} {j.get('error') or ''}")
        if jobs[0].get("stats"):
            st.json(jobs[0]["stats"])

    ents = requests.get(f"{API}/graph/entities", params={"limit": 50}, headers=HEADERS, timeout=30).json()
    if ents:
        st.subheader("Top entities")
        st.dataframe(ents, use_container_width=True)

        names = [e["name"] for e in ents]
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
                if msg.get("route_reason"):
                    cap += f" — {msg['route_reason']}"
                st.caption(cap)
            if msg.get("warning"):
                st.warning(msg["warning"])
            for c in msg.get("citations", []):
                label = f"[{c['n']}] {c.get('document_title') or c.get('document_id')}"
                if c.get("score") is not None:
                    label += f"  (score {c['score']:.2f})"
                if c.get("supported") is False:
                    label += " (unsupported)"
                with st.expander(label):
                    st.write(c["excerpt"])
                    if c.get("hop_path"):
                        st.graphviz_chart(_path_dot(c["hop_path"], c.get("edge_directions")))
                    for alt in c.get("alt_paths") or []:
                        st.caption("alt path: " + " → ".join(alt))
                    if c.get("years"):
                        st.caption("years: " + ", ".join(map(str, c["years"])))

    if q := st.chat_input("Ask a question (try a multi-hop one)"):
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Routing & retrieving…"):
                r = requests.post(
                    f"{API}/chat",
                    json={
                        "question": q,
                        "conversation_id": st.session_state.convo,
                        "mode": None if mode == "auto" else mode,
                    },
                    headers=HEADERS,
                    timeout=600,
                )
            if r.ok:
                data = r.json()
                st.session_state.convo = data["conversation_id"]
                st.markdown(data["answer"])
                cap = f"route: {data['route']}"
                if data.get("route_reason"):
                    cap += f" — {data['route_reason']}"
                st.caption(cap)
                if data.get("warning"):
                    st.warning(data["warning"])
                for c in data["citations"]:
                    label = f"[{c['n']}] {c.get('document_title') or 'source'}"
                    if c.get("score") is not None:
                        label += f"  (score {c['score']:.2f})"
                    if c.get("supported") is False:
                        label += " (unsupported)"
                    with st.expander(label):
                        st.write(c["excerpt"])
                        if c.get("hop_path"):
                            st.graphviz_chart(_path_dot(c["hop_path"], c.get("edge_directions")))
                        for alt in c.get("alt_paths") or []:
                            st.caption("alt path: " + " → ".join(alt))
                        if c.get("years"):
                            st.caption("years: " + ", ".join(map(str, c["years"])))
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "route": data["route"],
                        "route_reason": data.get("route_reason"),
                        "warning": data.get("warning"),
                        "citations": data["citations"],
                    }
                )
            else:
                st.error(r.text)
