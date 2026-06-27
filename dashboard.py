import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz

# Page config
st.set_page_config(
    page_title="AI Career Navigator",
    page_icon="🚀",
    layout="wide"
)

# Title
st.title("🚀 AI Career Navigator & Resume Intelligence System")
st.markdown("### Upload your resume and find the best matching jobs!")
st.divider()

# Skills list
SKILLS_LIST = [
    'python', 'sql', 'excel', 'power bi', 'tableau',
    'machine learning', 'data analysis', 'pandas', 'numpy',
    'matplotlib', 'statistics', 'r', 'spark', 'mysql',
    'postgresql', 'mongodb', 'data visualization', 'scikit-learn',
    'nlp', 'git', 'aws', 'azure', 'docker', 'linux',
    'communication', 'leadership', 'time management', 'dax',
    'tensorflow', 'keras', 'etl', 'bi tools', 'eda'
]

# Load job data
@st.cache_data
def load_jobs():
    df = pd.read_csv(r'C:\Users\KARAN DUBEY\career_navigator\data\clean_jobs.csv')
    return df.head(10000)

df_jobs = load_jobs()

# Sidebar
st.sidebar.title("📄 Resume Upload")
uploaded_file = st.sidebar.file_uploader("Upload your PDF Resume", type=['pdf'])

if uploaded_file is not None:
    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    resume_text = ""
    for page in doc:
        resume_text += page.get_text()

    st.sidebar.success("✅ Resume uploaded successfully!")

    # Extract skills
    resume_lower = resume_text.lower()
    found_skills = [skill for skill in SKILLS_LIST if skill in resume_lower]

    # ===============================
    # Row 1 - Metrics
    # ===============================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Total Jobs Analyzed", "10,000")
    col2.metric("✅ Skills Found", len(found_skills))
    col3.metric("❌ Skills to Learn", len(SKILLS_LIST) - len(found_skills))
    col4.metric("📊 Resume Length", f"{len(resume_text)} chars")

    st.divider()

    # ===============================
    # Row 2 - Job Matching
    # ===============================
    st.subheader("🎯 Top 10 Matching Jobs for You")

    with st.spinner("AI is matching your resume with jobs..."):
        job_descriptions = df_jobs['description'].fillna('').tolist()
        all_docs = [resume_text] + job_descriptions
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        top_10 = np.argsort(scores)[::-1][:10]

    results = []
    for rank, idx in enumerate(top_10, 1):
        results.append({
            'Rank': rank,
            'Job Title': df_jobs.iloc[idx]['title'],
            'Company': str(df_jobs.iloc[idx]['company_name']),
            'Location': str(df_jobs.iloc[idx]['location']),
            'Match Score': f"{scores[idx]*100:.1f}%"
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    st.divider()

    # ===============================
    # Row 3 - Skill Gap Analysis
    # ===============================
    st.subheader("📊 Skill Gap Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Skills You Already Have:**")
        for skill in found_skills:
            st.success(f"✓ {skill.upper()}")

    with col2:
        missing = [s for s in SKILLS_LIST[:15] if s not in found_skills]
        st.markdown("**❌ Skills You Need to Learn:**")
        for skill in missing[:7]:
            st.error(f"✗ {skill.upper()}")

    st.divider()

    # ===============================
    # Row 4 - Charts
    # ===============================
    st.subheader("📈 Skill Analysis Charts")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sizes = [len(found_skills), len(missing)]
        labels = ['Skills You Have', 'Skills to Learn']
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie(sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('Your Skill Match Overview')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        job_titles_top = [r['Job Title'][:25] for r in results[:5]]
        match_scores = [float(r['Match Score'].replace('%', '')) for r in results[:5]]
        ax2.barh(job_titles_top, match_scores, color='#3498db')
        ax2.set_xlabel('Match Score %')
        ax2.set_title('Top 5 Job Match Scores')
        st.pyplot(fig2)

    st.divider()

    # ===============================
    # Row 5 - Learning Resources
    # ===============================
    st.subheader("📚 Your Personalized Learning Roadmap")

    resources = {
        'aws': 'https://aws.amazon.com/training/',
        'azure': 'https://learn.microsoft.com/azure',
        'tableau': 'https://www.tableau.com/learn/training',
        'spark': 'https://spark.apache.org/docs/',
        'docker': 'https://docs.docker.com/get-started/',
        'machine learning': 'https://www.coursera.org/learn/machine-learning',
        'statistics': 'https://www.khanacademy.org/math/statistics-probability',
        'communication': 'https://www.coursera.org/courses?query=communication',
        'linux': 'https://www.linux.org/forums/#linux-tutorials.122'
    }

    for skill in missing[:7]:
        link = resources.get(skill, 'https://www.google.com/search?q=learn+' + skill)
        st.warning(f"❌ **{skill.upper()}** — [Click here for Free Resource]({link})")

else:
    st.info("👈 Please upload your PDF resume from the left sidebar!")
    st.markdown("""
    ### What does this tool do?
    - 📄 **Parses your resume** automatically using AI
    - 🎯 **Matches with 10,000+ jobs** using Machine Learning
    - 📊 **Identifies skill gaps** based on market demand
    - 📚 **Suggests free learning resources** to bridge the gap
    """)