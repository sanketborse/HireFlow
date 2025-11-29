import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


# ---------- PAGE CONFIG ----------
st.set_page_config(
    layout="wide",
    page_title="HireFlow",
    page_icon="📧",
)

# ---------- PREMIUM CSS ----------
premium_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* Global font */
html, body, [class*="st-"] {
    font-family: "Inter", sans-serif;
}

/* Center main content and limit width */
.main .block-container {
    max-width: 900px;
    padding-top: 4rem;
    margin: 0 auto;
}

/* Titles */
.app-title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.25rem;
}
.app-subtitle {
    text-align: center;
    font-size: 1.15rem;
    opacity: 0.8;
    margin-bottom: 2.5rem;
}

/* Card container */
.card {
    background: #111827cc;
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    border: 1px solid #4b5563;
    box-shadow: 0 18px 40px rgba(0,0,0,0.5);
}

/* Inputs */
.stTextInput > div > div > input {
    text-align: center;
    font-size: 1.05rem;
    padding: 0.6rem 0.5rem;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 1.05rem;
    margin-top: 0.8rem;
}

/* Headings */
h1, h2, h3, h4 {
    text-align: left;
}

/* Results spacing */
.results-separator {
    margin-top: 2.5rem;
    margin-bottom: 1.5rem;
}
</style>
"""
st.markdown(premium_css, unsafe_allow_html=True)


def create_streamlit_app(llm, portfolio, clean_text_fn):

    # ---------- HEADER ----------
    st.markdown("<div class='app-title'>📧 HireFlow</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>AI-Powered Outreach &amp; Lead Generation</div>",
        unsafe_allow_html=True,
    )

    # ---------- INPUT CARD ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    url_input = st.text_input(
        label="",
        value="",
        placeholder="Paste careers page URL here (e.g. https://company.com/careers)",
    )

    submit_button = st.button("Analyze Careers Page")

    st.markdown("</div>", unsafe_allow_html=True)  # close card

    # ---------- LOGIC + RESULTS ----------
    if submit_button:
        if not url_input.startswith(("http://", "https://")):
            st.error("Enter a valid URL starting with http:// or https://")
            return

        try:
            loader = WebBaseLoader(url_input)
            docs = loader.load()

            if not docs:
                st.error("Could not load content from this URL.")
                return

            raw_text = docs[0].page_content
            cleaned = clean_text_fn(raw_text)

            portfolio.load_portfolio()
            jobs = llm.extract_jobs(cleaned)

            if not jobs:
                st.warning("No jobs detected on this page.")
                return

            st.markdown("<div class='results-separator'><hr></div>", unsafe_allow_html=True)

            for idx, job in enumerate(jobs, start=1):
                st.markdown(f"### 🧩 Job #{idx}")
                st.json(job)

                # ------- Skill normalization -------
                skills = job.get("skills", [])

                if skills is None:
                    skills = []
                elif isinstance(skills, str):
                    skills = [skills]
                elif isinstance(skills, (int, float)):
                    skills = []
                elif isinstance(skills, list):
                    skills = [str(s) for s in skills]
                else:
                    skills = []

                # DEBUG
                print("DEBUG skills in main:", repr(skills), type(skills))

                # ------- Portfolio links -------
                links_meta = portfolio.query_links(skills)

                flat_links = []
                if links_meta:
                    for group in links_meta:
                        for meta in group:
                            link = meta.get("links")
                            if link:
                                flat_links.append(link)

                link_list_str = "\n".join(set(flat_links)) if flat_links else ""

                # ------- Email generation -------
                email = llm.write_mail(job, link_list_str)

                st.markdown("#### ✉ Generated Cold Email")
                st.code(email, language="markdown")

        except Exception as e:
            import traceback
            st.error(f"An Error Occurred: {e}")
            st.code(traceback.format_exc())


# Run app
if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
