import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Academic Copilot – Instructor Dashboard",
    layout="wide"
)

st.title("📊 Instructor Dashboard")
st.caption("Question clustering and confusion heatmap")

# Sidebar inputs
st.sidebar.header("Course Settings")
course_id = st.sidebar.text_input("Course ID", value="csci_demo")

if st.sidebar.button("Refresh data"):
    st.rerun()

# Fetch clusters
resp = requests.get(
    f"{API_BASE}/instructor/clusters",
    params={"course_id": course_id}
)

if resp.status_code != 200:
    st.error("Failed to fetch instructor data")
    st.stop()

data = resp.json()

st.subheader(f"Total questions: {data.get('total_questions', 0)}")
st.subheader(f"Detected clusters: {data.get('k', 0)}")

clusters = data.get("clusters", [])

if not clusters:
    st.warning("No data yet. Ask some student questions first.")
    st.stop()

# Build dataframe for charts
rows = []
for i, c in enumerate(clusters, start=1):
    rows.append({
        "cluster": f"Cluster {i}",
        "avg_confusion": c["avg_confusion"],
        "count": c["count"],
        "keywords": ", ".join(c["keywords"])
    })

df = pd.DataFrame(rows)

st.markdown("## 🔥 Confusion Heatmap (by cluster)")

fig, ax = plt.subplots()
ax.bar(df["cluster"], df["avg_confusion"])
ax.set_ylabel("Average confusion (0 to 1)")
ax.set_xlabel("Cluster")
ax.set_ylim(0, 1)
st.pyplot(fig)

st.markdown("## 📌 Question Volume (by cluster)")

fig2, ax2 = plt.subplots()
ax2.bar(df["cluster"], df["count"])
ax2.set_ylabel("Number of questions")
ax2.set_xlabel("Cluster")
st.pyplot(fig2)

st.markdown("## 🧠 Cluster Summary Table")
st.dataframe(df, use_container_width=True)


# Display clusters
for idx, cluster in enumerate(clusters, start=1):
    with st.expander(
        f"Cluster {idx} | "
        f"Avg confusion: {cluster['avg_confusion']} | "
        f"Questions: {cluster['count']}"
    ):
        st.markdown("**Top keywords:** " + ", ".join(cluster["keywords"]))

        st.markdown("**Sample questions:**")
        for q in cluster["questions"][:5]:
            st.write(f"• {q['question']}")
