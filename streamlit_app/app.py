import os

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="University GraphRAG Assistant", layout="wide")
st.title("University Knowledge Assistant (GraphRAG)")

page = st.sidebar.radio("Section", ["Chat", "Documents", "Graph Index"])

if page == "Documents":
    st.header("Upload documents")
    uploaded = st.file_uploader("pdf / docx / txt / md", type=["pdf", "docx", "txt", "md"])
    title = st.text_input("Title (optional)")
    if uploaded and st.button("Upload & chunk"):
        with st.spinner("Extracting, chunking, embedding…"):
            r = requests.post(
                f"{API}/documents",
                files={"file": (uploaded.name, uploaded.getvalue())},
                params={"title": title or uploaded.name},
                timeout=300,
            )
        if r.ok:
            st.success(f"Uploaded {r.json()['title']} → status {r.json()['status']}")
        else:
            st.error(r.text)

    st.subheader("Indexed documents")
    docs = requests.get(f"{API}/documents", timeout=30).json()
    for d in docs:
        st.write(f"- **{d['title']}** — `{d['status']}` ({d['filename']})")

elif page == "Graph Index":
    st.header("Knowledge graph")
    stats = requests.get(f"{API}/graph/stats", timeout=30).json()
    cols = st.columns(6)
    for col, key in zip(
        cols,
        ["entities", "relationships", "communities", "text_units", "community_reports"], strict=False,
    ):
        col.metric(key.replace("_", " "), stats.get(key, 0))
    st.write(f"Indexed: **{stats.get('indexed')}**")

    if st.button("Run GraphRAG index"):
        r = requests.post(f"{API}/ingest", timeout=30)
        if r.ok:
            st.info(f"Job {r.json()['id']} started — refresh to see progress")
        else:
            st.error(r.text)

    jobs = requests.get(f"{API}/ingest", timeout=30).json()
    if jobs:
        st.subheader("Jobs")
        for j in jobs[:5]:
            st.write(f"- `{j['status']}` {j['created_at']} {j.get('error') or ''}")
        if jobs[0].get("stats"):
            st.json(jobs[0]["stats"])

    ents = requests.get(f"{API}/graph/entities", params={"limit": 50}, timeout=30).json()
    if ents:
        st.subheader("Top entities")
        st.dataframe(ents, use_container_width=True)

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
                st.caption(f"route: {msg['route']}")
            for c in msg.get("citations", []):
                with st.expander(f"[{c['n']}] {c.get('document_title') or c.get('document_id')}"):
                    st.write(c["excerpt"])
                    if c.get("hop_path"):
                        st.caption("hop path: " + " → ".join(c["hop_path"]))

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
                    timeout=600,
                )
            if r.ok:
                data = r.json()
                st.session_state.convo = data["conversation_id"]
                st.markdown(data["answer"])
                st.caption(f"route: {data['route']}")
                for c in data["citations"]:
                    with st.expander(f"[{c['n']}] {c.get('document_title') or 'source'}"):
                        st.write(c["excerpt"])
                        if c.get("hop_path"):
                            st.caption("hop path: " + " → ".join(c["hop_path"]))
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "route": data["route"],
                        "citations": data["citations"],
                    }
                )
            else:
                st.error(r.text)
