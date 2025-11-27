import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

    def extract_jobs(self, cleaned_text):
        """
        Ask the LLM to extract job postings from the cleaned careers page text.
        Returns a list of job dicts: {role, experience, skills, description}.
        """
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The scraped text is from the careers page of a website.
            Your job is to extract the job postings and return them in **valid JSON** format.
            Every job posting object must have the following keys:

            - "role": string
            - "experience": string
            - "skills": list of strings
            - "description": string

            If there are multiple roles, return a JSON array of objects.
            If there is only one role, still return it as a JSON array with one object.
            Only return the valid JSON. No explanation, no markdown, no comments.

            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            parsed = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big or output not valid JSON.")

        # Ensure result is always a list of job dicts
        return parsed if isinstance(parsed, list) else [parsed]

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
            You are **Sanket Borse**, a Business Development Executive at **Accenture**.
            Accenture is an AI & Software Consulting company dedicated to facilitating
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
