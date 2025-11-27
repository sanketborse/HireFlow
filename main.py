import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


# Always set page config as early as possible
st.set_page_config(
    layout="wide",
    page_title="Cold Email Generator",
    page_icon="📧",
)


def create_streamlit_app(llm: Chain, portfolio: Portfolio, clean_text_fn):
    st.title("📧 HireFlow")
    st.write("Enter the URL of the company's careers page")

    url_input = st.text_input(
        "Enter a URL:",
        value="",
        placeholder="e.g. https://company.com/careers",
    )

    submit_button = st.button("Submit")

    if submit_button:
        # Basic URL validation
        if not url_input or not url_input.startswith(("http://", "https://")):
            st.error("Please enter a valid URL starting with http:// or https://")
            return

        try:
            # Load page content
            loader = WebBaseLoader(url_input)
            docs = loader.load()

            if not docs:
                st.error("No content could be loaded from the given URL.")
                return

            raw_text = docs[0].page_content
            cleaned = clean_text_fn(raw_text)

            # Load portfolio into ChromaDB
            portfolio.load_portfolio()

            # Extract jobs using LLM
            jobs = llm.extract_jobs(cleaned)

            if not jobs:
                st.warning("No job postings could be extracted from this page.")
                return

            for idx, job in enumerate(jobs, start=1):
                st.markdown(f"### 🧩 Job #{idx}")
                st.json(job)

                # Get skills safely
                skills = job.get("skills", [])

                # Normalize skills before querying portfolio
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

                links = portfolio.query_links(skills)

                # Format links for email prompt
                link_list_str = ""
                if links:
                    # links is a list of list of metadatas, flatten it
                    flat_links = []
                    for group in links:
                        for meta in group:
                            link = meta.get("links")
                            if link:
                                flat_links.append(link)
                    if flat_links:
                        link_list_str = "\n".join(set(flat_links))

                email = llm.write_mail(job, link_list_str)

                st.markdown("#### ✉ Generated Cold Email")
                st.code(email, language="markdown")

        except Exception as e:
            import traceback

            st.error(f"An Error Occurred: {e}")
            st.code(traceback.format_exc())


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()  # uses default CSV path
    create_streamlit_app(chain, portfolio, clean_text)
