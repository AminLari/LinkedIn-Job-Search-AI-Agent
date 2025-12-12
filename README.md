# AI Job Application Assistant Using n8n & Mistral-3B

## Overview
This project implements an automated job-application assistant using **n8n**, **Mistral-3 3B**, and the **GitHub API**.  
The system searches for job postings on LinkedIn, customizes a user-provided cover-letter template for each job, generates tailored demo projects to showcase the applicant’s skills, uploads these projects to GitHub, and emails the final cover letters to the user.

<p> <img src="https://github.com/user-attachments/assets/4f2a87e9-97dd-4516-bd7a-ae4ff723ddc5" width="1000"> </p> 

## Features
- **Automated Job Search:** Scrapes or fetches LinkedIn job listings via API.
- **AI-Generated Cover Letters:** Uses Mistral-3 3B to personalize cover letters for each role.
- **Demo Project Generation:** Builds a job-specific demo project demonstrating relevant skills.
- **GitHub Upload:** Publishes the generated demo project to the user’s GitHub repository.
- **Cover Letter Export:** Converts the generated cover letters into `.txt` files.
- **Email Delivery:** Sends the cover-letter files to the user’s email.
- **Fully Automated Workflow:** Powered by n8n's scheduled triggers and modular nodes.

## File Structure
- `workflow.json` — Exported n8n workflow.
- `docs/` — Agent-generated demo projects.
- `AI Agent Workflow.jpg` — Workflow diagram.
- `README.md` — Documentation for the repository.

## Installation

### Prerequisites
Make sure you have the following:
- **n8n** (self-hosted or Cloud)
- **GitHub Personal Access Token** (with repo permissions)
- **API access** to LinkedIn job-scraping (e.g., Apify Actor)
- **Mistral-3 3B API key**
- **SMTP/Gmail credentials** for sending emails

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/LinkedIn-Job-Search-AI-Agent.git
   cd LinkedIn-Job-Search-AI-Agent

2. **Import the workflow into n8n:**
   Open n8n → Import Workflow
   Upload workflow.json
3. **Configure required credentials in n8n:**
   LinkedIn / Apify API key
   Mistral-3 3B API key
   GitHub PAT
   Email service (SMTP / Gmail)
4. **Set your cover-letter template:**
   Link your cover letter template in the “Rewrite Cover Letter” node.

## Usage

1. **Provide your base cover-letter template**  
   Place your general-purpose cover letter inside the “Rewrite Cover Letter” node in n8n.

2. **Run the workflow manually or schedule it**  
   Use n8n’s Cron node to execute the pipeline daily or weekly.

3. **Automated pipeline steps**  
   Once triggered, the workflow will:
   - Fetch job postings from LinkedIn or your scraper API  
   - Extract the job title, company name, and job details  
   - Generate a personalized cover letter using Mistral-3 3B  
   - Create a tailored demo project related to the role  
   - Upload the generated project files to your GitHub repository  
   - Convert the cover letter into a `.txt` file  
   - Email the final cover letter and GitHub link to the user  

4. **Review your results**
   - **Inbox:** Each job produces a personalized `.txt` cover letter  
   - **GitHub:** A new demo project is automatically uploaded for every job  

## Results

The system automatically produces the following for each job posting:

- A **uniquely tailored cover letter**  
- A **job-specific demo project** showcasing relevant skills  
- An **uploaded GitHub repository** for each generated project  
- An **emailed `.txt` file** containing the customized cover letter  

These outputs significantly strengthen job applications by increasing personalization and demonstrating hands-on ability.


## 📞 Contact

If you have any questions, ideas, or want to contribute, feel free to reach out:

- **Name:** Mark Lari  
- **Email:** [mark.lari.work@gmail.com](mailto:mark.lari.work@gmail.com)  
- **GitHub:** [AminLari](https://github.com/aminlari)

Contributions, issues, and pull requests are always welcome!
