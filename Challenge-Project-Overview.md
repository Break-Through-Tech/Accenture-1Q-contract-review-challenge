---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | Project utilizes standard libraries (HuggingFace transformers, scikit-learn, pandas) compatible with free-tier Google Colab environments. |
| Data Readiness | 🟡 | CUAD dataset is well-structured but requires significant effort to parse raw PDF text and align token-level labels for 41 categories, which may consume excessive time during early weeks. |
| Resource Check | 🟢 | Project fits within memory constraints; however, fine-tuning large transformer models will require careful batch size management in Colab. |

### Internal Scores
- **Student Fit Score:** 7/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
The CUAD-based contract triage pipeline is a high-value industrial use case with excellent potential for technical rigor. To ensure success within the timeframe of the program, I suggest: (i) pivoting from fine-tuning from scratch to utilizing pre-trained lightweight models (e.g., DistilRoBERTa) to reduce hardware overhead demand; and (ii) constraining the scope by focusing on the 10 most impactful clause categories, and having the full 41 as a stretch goal.

---

# Contract Review Challenge

**Company / Org:** Accenture  
**Challenge Advisor:** Adarsh Ravikumar, adarsh.ravikumar@accenture.com  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Accenture
Accenture is a leading global professional services company that provides a broad range of services and solutions in strategy, consulting, technology, and operations. The team objective for this project is to modernize legal operations by leveraging artificial intelligence to increase the efficiency and accuracy of contract review processes.

---

## 🎯 The Challenge
### Project Summary
This project involves building a pipeline to automatically detect key clauses in commercial contracts and flag them by risk level (Low, Medium, High) to generate a consolidated triage score. By utilizing the CUAD dataset and applying NLP strategies like multi-label classification and explainable risk-scoring, the team will help legal and procurement departments prioritize high-risk documents. This effort directly addresses the manual labor bottleneck currently faced by teams reviewing thousands of contracts annually.

### Success Criteria
For clause detection: per-category precision/recall/F1. For risk scoring: Spearman correlation and bucket agreement between the model's risk rankings and the advisor's hand-ranked clauses.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Exploration & Preprocessing | Parsing raw CUAD PDFs, text normalization, and establishing token-level label alignment strategies for the target 41 categories. |
| **October** | Feature Engineering & Baseline Modeling | Implementing chunk-based multi-label classification architectures and establishing baseline performance benchmarks using standard transformer encoders. |
| **November** | Model Optimization & Evaluation | Conducting hyperparameter tuning, implementing the rule-based risk-scoring layer, and executing model validation against advisor-provided ground truth. |
| **December** | Insights, Deliverables & Presentation | Finalizing the end-to-end pipeline, calculating reviewer time-saved metrics, and preparing the final technical presentation for stakeholders. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** CUAD Dataset (Contract Understanding Atticus Dataset)  
**Format:** JSON, Raw Text/PDF  
**Size:** under 1gb  
**Location:** [Link to be provided by Advisor during kickoff]  

### Key Details
- Real-world commercial contracts from the CUAD dataset (510 contracts, 41 expert-annotated clause categories), raw text/PDF available.
- Teams must implement strict preprocessing rules to handle document length variance and ensure text cleaning captures the necessary legal terminology for high-accuracy classification.

---

## 🛠️ Suggested Approach
**ML Problem Type:** NLP & Classification  
**Recommended Libraries:** HuggingFace Transformers, PyTorch/TensorFlow, Scikit-learn, Pandas  
**Evaluation Metrics:** Precision, Recall, F1-Score for classification; Spearman Correlation for risk-ranking alignment.

---

## 📚 Resources to Get Started
The following resources will help your team understand the problem space and potential technical approaches for this project:
**Background Reading:**
- The CUAD Paper: Atticus Project research on automated legal document review.
**Technical Tutorials:**
- HuggingFace NLP Course: Fine-tuning models for token classification.
**Code Examples:**
- CUAD GitHub repository and standard transformer boilerplate for text classification tasks.

---

## 🤝 How We'll Work Together
**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Email and scheduled Lab Section syncs  
**Response time:** 48 business hours  
**Recommended Tools:**
- **Coding:** Google Colab Free Tier  
- **Collaboration:** GitHub, Notion  
- **Virtual Meetings:** Zoom, Google Meet  

---

## 🚀 Getting Started
1. **Review this overview document** and note any questions for our first meeting.
2. **Begin reviewing the dataset** using the link provided in the Dataset section.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects).

I'm excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
