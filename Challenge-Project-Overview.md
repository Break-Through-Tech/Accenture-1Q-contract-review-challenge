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
**Challenge Advisor:** Gopinath Kolluru, [Email address]  
**AI Studio Coach:** Parth Dali, parth.dali@breakthroughtech.org  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Accenture
Accenture is a leading global professional services company that provides a broad range of services and solutions in strategy, consulting, technology, and operations. 

---

## 🎯 The Challenge
### Project Summary
In this project, you will use real-world commercial contracts from the CUAD dataset (510 contracts, 41 expert-annotated clause categories) and NLP techniques including chunk-based multi-label classification with fine-tuned transformer encoders, paired with an explainable rule-based risk-scoring layer, to build a pipeline that automatically detects key clauses, flags them as Low/Medium/High risk, and rolls these up into a contract-level triage score. This will help our company address the bottleneck legal and procurement teams face when manually reviewing tens of thousands of contracts a year to find the small number of clauses that carry meaningful risk, enabling reviewers to prioritize which contracts to open first.

### Success Criteria
Success has two tracks:
- For clause detection: per-category precision/recall/F1 clearly beating the baseline (accuracy is misleading under CUAD's imbalance), with error analysis on where the model struggles.   
- For risk scoring: since there are no ground-truth labels, success means strong Spearman correlation and bucket agreement between the model's risk rankings and the advisor's hand-ranked clauses, plus a sensitivity analysis showing the High/Medium boundary is stable.

Overall, a successful December outcome is a working end-to-end pipeline producing risk-scored clause registers the advisor finds plausible and useful, a clean documented repo, and a final report covering results, limitations, and estimated reviewer time saved — an auditable triage tool the advisor would actually trust, not a black box.

### Stretch Goals
Stretch goals include span extraction, a trained risk model benchmarked against the rule-based baseline, LLM-generated clause explanations, broader category coverage, a Streamlit/Gradio demo, and an active-learning loop using advisor/model disagreements. These extend modeling or usability without affecting core deliverables.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Analysis, Design & Build | Clean and split the CUAD data, run EDA on class imbalance, build a chunking strategy, and establish a TF-IDF/keyword baseline with per-category metrics. |
| **October** | Build & Refine | Fine-tune a transformer encoder for multi-label clause classification, address class imbalance, evaluate with per-category precision/recall/F1, and conduct error analysis. |
| **November** | Refine & Readiness | Build and calibrate the four-signal risk-scoring layer, assemble the end-to-end pipeline, and validate risk rankings against advisor-labeled examples. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** CUAD Dataset (Contract Understanding Atticus Dataset)  
**Format:** JSON, Raw Text/PDF  
**Size:** under 1gb  
**Location:** https://github.com/TheAtticusProject/cuad  

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
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* Email me copy your teammates and AI Studio Coach
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]


**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
