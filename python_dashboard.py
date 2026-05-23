# python_dashboard.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================
# Load dataset
# ==========================
file_path = r"E:\internship all\Student Learning Platform\dummy_student_dataset_encoded.xlsx"
df = pd.read_excel(file_path)

# ==========================
# Streamlit Layout
# ==========================
st.set_page_config(page_title="Student Dashboard", layout="wide")
st.title("📊 Student Learning Dashboard")
st.markdown("This dashboard provides insights, visualizations, and early warning alerts for student learning performance.")

# ==========================
# Dataset Overview
# ==========================
st.subheader("🔹 Dataset Overview")
st.write(df.head())
st.write(f"Dataset Shape: {df.shape}")

# ==========================
# 1. Distribution Plots (Numeric Features)
# ==========================
st.subheader("🔹 Numeric Feature Distributions")
numeric_cols = [
    'marks_obtained', 'completion_pct', 'study_hours_per_week',
    'time_spent_hours', 'confidence_score', 'videos_watched',
    'quizzes_attempted', 'peer_interaction_score', 'self_assessment_score'
]

for col in numeric_cols:
    if col in df.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df[col], bins=30, kde=True, color="skyblue", ax=ax)
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.1f}")
        ax.set_title(f"{col.replace('_',' ').title()} Distribution")
        ax.set_xlabel(col.replace('_',' ').title())
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        st.write(f"📊 Insight: {col.replace('_',' ').title()} ranges from {df[col].min()} to {df[col].max()}, median = {median_val}, mean = {mean_val:.1f}")

# ==========================
# 2. Patterns Across Courses, Topics, Learning Style
# ==========================
st.subheader("🔹 Patterns by Course, Topic, and Learning Style")
group_cols = ['course_name', 'topic_name', 'preferred_learning_style']

for col in group_cols:
    if col in df.columns:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.violinplot(x=col, y='marks_obtained', data=df, palette="Set2", ax=ax)
        sns.swarmplot(x=col, y='marks_obtained', data=df, color="k", alpha=0.4, ax=ax)
        ax.set_title(f"Marks Obtained by {col.replace('_',' ').title()}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        medians = df.groupby(col)['marks_obtained'].median()
        st.write(f"📊 Median marks by {col.replace('_',' ').title()}:\n", medians)

# ==========================
# 3. Segmentation by Demographics
# ==========================
st.subheader("🔹 Demographic Analysis")
demo_cols = ['age', 'region', 'education_level']

for col in demo_cols:
    if col in df.columns:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(x=col, y='marks_obtained', data=df, palette="Pastel1", ax=ax)
        sns.stripplot(x=col, y='marks_obtained', data=df, color="k", alpha=0.4, ax=ax)
        ax.set_title(f"Marks Obtained by {col.replace('_',' ').title()}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        medians = df.groupby(col)['marks_obtained'].median()
        st.write(f"📊 Median marks by {col.replace('_',' ').title()}:\n", medians)

# ==========================
# 4. Correlation Heatmap
# ==========================
st.subheader("🔹 Correlation Analysis")
corr = df.corr(numeric_only=True)
mask = (corr >= 0.4) | (corr <= -0.4)
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(corr.where(mask), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap of Numeric Features (|corr| >= 0.4)")
st.pyplot(fig)

top_corr = corr.unstack().sort_values(ascending=False)
top_corr = top_corr[(top_corr < 1.0) & ((top_corr >= 0.5) | (top_corr <= -0.5))]
st.write("📊 Strong Correlations (|corr| >= 0.5):", top_corr)

# ==========================
# 5. K-Means Clustering
# ==========================
st.subheader("🔹 Student Clustering (K-Means)")
cluster_features = [
    'time_spent_hours', 'videos_watched', 'quizzes_attempted',
    'interactive_exercises_done', 'forum_posts_count', 'marks_obtained',
    'completion_pct'
]
X = df[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_opt = 3
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(x='time_spent_hours', y='marks_obtained', hue='cluster', palette='Set1', data=df, s=80, ax=ax)
ax.set_title("Student Clusters: Engagement vs Marks")
ax.set_xlabel("Time Spent Hours")
ax.set_ylabel("Marks Obtained")
st.pyplot(fig)

st.write("📊 Cluster Insights:")
for cluster_num in range(k_opt):
    cluster_data = df[df['cluster']==cluster_num]
    avg_marks = cluster_data['marks_obtained'].mean()
    avg_time = cluster_data['time_spent_hours'].mean()
    avg_completion = cluster_data['completion_pct'].mean()
    st.write(f"**Cluster {cluster_num}:** {len(cluster_data)} students, Avg Marks: {avg_marks:.1f}, Avg Study Hours: {avg_time:.1f}, Avg Completion: {avg_completion:.2f}")

# ==========================
# 6. At-Risk Students Analysis
# ==========================
st.subheader("⚠️ At-Risk Students")
engagement_cols = ['time_spent_hours', 'videos_watched', 'quizzes_attempted', 'interactive_exercises_done', 'forum_posts_count']
marks_threshold = df['marks_obtained'].median()
df['low_engagement'] = df[engagement_cols].lt(df[engagement_cols].median()).sum(axis=1) >= 3
df['low_performance'] = df['marks_obtained'] < marks_threshold
df['at_risk'] = df['low_engagement'] & df['low_performance']

st.write(f"Total At-Risk Students: {df['at_risk'].sum()} / {len(df)}")

# At-Risk per Course
course_completion = df.groupby('course_name').agg({
    'completion_pct':'mean',
    'at_risk':'sum',
    'student_id':'count'
}).reset_index()
course_completion['at_risk_percentage'] = course_completion['at_risk'] / course_completion['student_id'] * 100

fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x='course_name', hue='at_risk', data=df, palette={True:'red', False:'green'}, ax=ax)
ax.set_title("At-Risk Students per Course (Red=At-Risk, Green=Safe)")
ax.set_xlabel("Course Name")
ax.set_ylabel("Number of Students")
st.pyplot(fig)

# Early Warning Alerts
st.subheader("📌 Early Warning Alerts")
for course in course_completion['course_name']:
    course_data = course_completion[course_completion['course_name']==course].iloc[0]
    if course_data['at_risk_percentage'] > 20:
        st.warning(f"{course}: {course_data['at_risk_percentage']:.1f}% at-risk students. Consider intervention.")

st.success("✅ Dashboard fully loaded!")