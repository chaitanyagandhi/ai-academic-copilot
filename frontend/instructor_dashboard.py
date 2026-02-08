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
lecture_id = st.sidebar.text_input("Lecture ID (optional)", value="")
lecture_id = lecture_id.strip() or None

if st.sidebar.button("Refresh data"):
    st.rerun()

# Fetch clusters
base_params = {"course_id": course_id}
if lecture_id:
    base_params["lecture_id"] = lecture_id

resp = requests.get(
    f"{API_BASE}/instructor/clusters",
    params=base_params
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

st.markdown("## ⏱️ Confusion Over Time")

trend_resp = requests.get(
    f"{API_BASE}/instructor/confusion_trend",
    params=base_params
)

if trend_resp.status_code == 200:
    trend_data = trend_resp.json().get("points", [])
    if trend_data:
        tdf = pd.DataFrame(trend_data)
        tdf["time"] = pd.to_datetime(tdf["time"])

        fig3, ax3 = plt.subplots()
        ax3.plot(tdf["time"], tdf["avg_confusion"], marker="o")
        ax3.set_ylabel("Average confusion")
        ax3.set_xlabel("Time")
        ax3.set_ylim(0, 1)
        st.pyplot(fig3)
    else:
        st.info("Not enough data yet to show confusion trend.")
else:
    st.warning("Failed to load confusion trend.")

st.markdown("## 🚨 Alerts")
alerts_resp = requests.get(
    f"{API_BASE}/instructor/alerts",
    params=base_params
)
if alerts_resp.status_code == 200:
    alerts_data = alerts_resp.json()
    current_alerts = alerts_data.get("alerts", [])
    history = alerts_data.get("history", [])

    if current_alerts:
        for a in current_alerts:
            st.error(f"{a.get('message')} (severity: {a.get('severity')})")
    else:
        st.info("No active alerts.")

    if history:
        st.markdown("**Recent alert history:**")
        for h in history[:5]:
            st.write(f"- {h.get('message')} (severity: {h.get('severity')})")
else:
    st.warning("Failed to load alerts.")

st.markdown("## ✅ Teaching Recommendations")
rec_resp = requests.get(
    f"{API_BASE}/instructor/recommendations",
    params=base_params
)
if rec_resp.status_code == 200:
    rec_data = rec_resp.json()
    st.write(rec_data.get("recommendations", "No recommendations yet."))
else:
    st.warning("Failed to load recommendations.")


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
