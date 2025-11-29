import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


# ---------- PREMIUM PAGE CONFIG ----------
st.set_page_config(
    layout="wide",
    page_title="HireFlow",
    page_icon="📧"
)

# ---------- CUSTOM PREMIUM CSS ----------
premium_css = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: "Inter", sans-serif;
}

.center-container {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

.card {
    background: #ffffff15;
    padding: 2.5rem;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    border: 1px solid #ffffff25;
    width: 60%;
    margin-top: 40px;
    box-shadow: 0 0 25px rgba(0,0,0,0.12);
}

.stTextInput > div > div > input {
    text-align: center;
    font-size: 1.1rem;
}

.stButton>button {
    width: 50%;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 1.1rem;
    margin: 0 auto;
    display: block;
}

h1, h2, h3, h4 {
    text-align: center !important;
}

</style>
"""

st.markdown(premium_css, unsafe_allow_html=True)


def create_streamlit_app(llm, portfolio, clean_text_fn):

    st.markdown("<div class='center-container'>", unsafe_allow_html=True)

    st.markdown(
        "<h1 style='font-size:3rem; font-weight:700; margin-top:20px;'>📧 HireFlow</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:1.3rem; opacity:0.8; margin-top:-10px;'>AI-Powered Outreach & Lead Generation</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # ---------------- UI INPUT ----------------
    url_input = st.text_input(
        "",
        value="",
        placeholder="Paste careers page URL here (e.g. https://company.com/careers)",
    )

    submit_button = st.button("Analyze Careers Page")

    # ---------------- LOGIC ----------------
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

            st.markdown("<hr>", unsafe_allow_html=True)

            for idx, job in enumerate(jobs, start=1):

                st.markdown(f"<h3>🧩 Job #{idx}</h3>", unsafe_allow_html=True)
                st.json(job)

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

                print("DEBUG skills in main:", repr(skills), type(skills))

                links_meta = portfolio.query_links(skills)

                flat_links = []
                if links_meta:
                    for group in links_meta:
                        for meta in group:
                            link = meta.get("links")
                            if link:
                                flat_links.append(link)

                link_list_str = "\n".join(set(flat_links)) if flat_links else ""

                email = llm.write_mail(job, link_list_str)

                st.markdown("<h4>✉ Generated Cold Email</h4>", unsafe_allow_html=True)
                st.code(email, language="markdown")

        except Exception as e:
            import traceback
            st.error(f"An Error Occurred: {e}")
            st.code(traceback.format_exc())

    st.markdown("</div>", unsafe_allow_html=True)      # close card
    st.markdown("</div>", unsafe_allow_html=True)      # close center-container


# Run app
if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
