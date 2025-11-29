import os
import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
        )

    def extract_jobs(self, cleaned_text: str):
        """
        Ask the LLM to extract job postings from the cleaned careers page text.
        Returns a list of job dicts: {role, experience, skills, description}.
        """

        # 🔹 1) Limit context size so we don't overload the model
        MAX_CHARS = 8000  # you can tweak this if needed
        if len(cleaned_text) > MAX_CHARS:
            cleaned_text = cleaned_text[:MAX_CHARS]

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The scraped text is from the careers page of a website.
            Your job is to extract the job postings and return them in valid JSON format.

            Rules:
            - ALWAYS return a JSON ARRAY, even if there's only one job.
            - Each job object MUST have the following keys:
                - "role": string
                - "experience": string
                - "skills": list of strings
                - "description": string
            - Even if the information is incomplete, still return a best-effort list of jobs.
            - Do NOT include any explanation, comments, markdown, or text before/after the JSON.
            - The output must be directly parseable by json.loads().

            ### VALID JSON ARRAY ONLY (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        raw = res.content

        # Debug log (Render logs)
        print("=== RAW LLM JOB OUTPUT (first 800 chars) ===")
        print(raw[:800])

        # 🔹 2) Try direct JSON parse
        try:
            jobs = json.loads(raw)
        except json.JSONDecodeError:
            # 🔹 3) Fallback: try to extract the JSON array from messy output
            try:
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if not match:
                    raise OutputParserException("No JSON array found in LLM output.")
                json_str = match.group(0)
                jobs = json.loads(json_str)
            except Exception as e:
                print("=== JSON PARSE FAILED ===")
                print(str(e))
                raise OutputParserException(
                    "Context too big or output not valid JSON."
                )

        # 🔹 4) Normalize to list
        if isinstance(jobs, dict):
            jobs = [jobs]
        elif not isinstance(jobs, list):
            raise OutputParserException("LLM output is not a JSON array.")

        print("=== PARSED JOB COUNT ===", len(jobs))
        return jobs

    def write_mail(self, job, links: str):
        """
        Generate a cold email based on the job description and portfolio links.
        """
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### PORTFOLIO LINKS:
            {link_list}

            ### INSTRUCTION:
            You are **Alex Hardy**, a Business Development Executive at **Apex Systems Consulting**.
            Apex Systems Consulting is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools.
            Over your experience, you have empowered numerous enterprises with tailored solutions,
            fostering scalability, process optimization, cost reduction, and heightened efficiency.

            Your job is to write a professional cold email to the client regarding the job mentioned above,
            explaining how Accenture can fulfill their needs.

            - Use a polite, concise, and impactful tone.
            - Add the most relevant ones from the provided links to showcase Accenture's portfolio.
            - Do NOT include any preamble or explanation around the email.
            - Output should be ready-to-send email text (with subject line).

            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke(
            {
                "job_description": str(job),
                "link_list": links or "No portfolio links available",
            }
        )
        return res.content


if __name__ == "__main__":
    # Debug: check if key is loaded (don't print it)
    print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))
