# Contract Review Challenge

**Company / Org:** Accenture  
**Challenge Advisor:** Gopinath Kolluru, gopinath.kolluru@accenture.com  
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
**Location:** [Data folder](data/cuad)

### Key Details
- `CUADv1.json` contains 510 commercial contracts and 13,823 annotated answer spans across 41 contract-review categories.
- Use the prepared JSON files: `train_separate_questions.json` contains 408 contracts and `test.json` contains 102 contracts. These are the **official** train/test splits released by The Atticus Project — split at the contract level (not by individual clause) to prevent data leakage, and directly comparable to the results in the original CUAD paper. Do not re-split the data yourselves.
- Contract text is already available in each JSON document's `paragraphs[].context` field, with clause questions in `paragraphs[].qas[]` and labeled spans in `paragraphs[].qas[].answers[]`. **Do not parse raw PDFs for this project.**
- `category_descriptions.csv` provides the name, description, answer format, and group for each of the 41 categories.
- Contracts vary substantially in length, so teams should develop a chunking strategy, preserve important legal terminology during cleaning, and account for class imbalance.

| Dataset / Source | Purpose in Project | Format | Access |
|---|---|---|---|
| **CUAD Category Descriptions** | Defines the 41 clause categories and provides guidance on what each category represents. Useful for building the label mapping and understanding the classification task. | CSV | [CUAD GitHub Repository](https://github.com/TheAtticusProject/cuad) |
| **CUAD Dataset – Hugging Face** | Provides a machine-learning-friendly way to load CUAD directly into Python and Hugging Face workflows.| Hugging Face Dataset | [CUAD on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa) |

> ⚠️ **Note on Hugging Face naming:** use `theatticusproject/cuad-qa` specifically. The similarly named `theatticusproject/cuad` (no `-qa`) is a different, unstructured repository containing only documentation text — it is **not** usable contract data.

---

## 🛠️ Suggested Approach
**ML Problem Type:** NLP & Classification  
**Recommended Libraries:** HuggingFace Transformers, PyTorch/TensorFlow, Scikit-learn, Pandas  
**Algorithm Examples:** TF-IDF/keyword baselines and a lightweight pre-trained transformer encoder such as DistilRoBERTa  
**Suggested Pipeline:** Contract Text → Preprocessing & Chunking → Multi-label Clause Classification → Evidence Extraction → Rule-based Risk Scoring → Contract-level Triage Score  
**Evaluation Metrics:** Precision, Recall, F1-Score for classification; Spearman Correlation for risk-ranking alignment.  
**Development Environment:** Google Colab for model training and experiments; VS Code and Jupyter Notebooks for development and analysis.

---

## 📚 Resources to Get Started

These resources will help your team understand the CUAD dataset, legal clause classification, transformer models, and the overall project approach.

**Background Reading:**
- [CUAD – Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad/) — Dataset overview and legal contract review problem.
- [CUAD Labeling Handbook](https://www.atticusprojectai.org/labeling-handbook/) — Definitions and examples of the 41 clause categories.
- [CUAD Research Paper](https://arxiv.org/abs/2103.06268) — Technical background on the CUAD dataset.

**Technical Tutorials:**
- [Hugging Face – Text Classification](https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification) — Fine-tuning transformer models.
- [Hugging Face – Datasets](https://huggingface.co/docs/datasets/) — Loading and processing datasets.
- [Hugging Face padding and truncation guide](https://huggingface.co/docs/transformers/main/pad_truncation)

**Code & Data:**
- [Official CUAD GitHub Repository](https://github.com/TheAtticusProject/cuad) — Reference code and dataset resources.
- [Guide to the provided data files](data/cuad/README.md)

**Other:**
- [scikit-learn precision, recall, and F-score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)

**Recommended Tools:**
- **Python:** pandas, NumPy, scikit-learn
- **ML/NLP:** Hugging Face Transformers, Hugging Face Datasets, PyTorch
- **Development:** Google Colab, VS Code, Jupyter Notebooks
- **Data Analysis:** pandas, NumPy, scikit-learn
- **Collaboration:** Git, GitHub Projects, Notion
- **Documentation:** GitHub README and project documentation
- **Virtual Meetings:** Zoom, Google Meet

Feel free to explore beyond these, and share anything interesting you find with me!

> **Tip:** Start with the CUAD dataset and Labeling Handbook before selecting a model. You are encouraged to explore additional tools, techniques, and resources as you develop the project and share useful findings with the team.

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

**Other ways to reach out to me with questions:**  

**Communication:** Email (gopinath.kolluru@accenture.com); please copy your teammates and AI Studio Coach. 

* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

---

## 🚀 Getting Started

Read this overview and list your open questions before our first team meeting.

1. **Review this overview document** and note any questions for our first meeting: Understand the project goals, technical approach, dataset expectations, milestones.
2. **Begin reviewing the JSON dataset** in the [data folder](data/cuad) Explore the 41 clause categories and identify the initial subset of contracts you will use for data exploration and baseline development.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)
4. **Prepare Open Questions:** Record questions, assumptions, and areas where you need clarification before the first team meeting.
5. **Document Your Decisions:** Keep important technical decisions and findings in GitHub Issues or project documentation so the entire team can follow the project's progress.

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
