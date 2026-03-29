# COMPAS Recidivism — Explainability & Interpretability as Diagnostic Tools

**DNSC 6330: Responsible Machine Learning**  
Individual Homework 2

> **Generative AI Disclosure:** Generative AI tools were used as a learning aid during the development of this work — specifically for brainstorming code structure, translating R syntax to Python, and reviewing outputs for accuracy. All AI-generated content was critically reviewed, validated, and integrated as the author's own intellectual product. This disclosure is made in accordance with GW's Generative AI Use Policy.

## Purpose

This repository contains the Individual Homework 2 submission for DNSC 6330, building directly on the Lecture 02 live-coding exercise. It extends the COMPAS recidivism analysis from Homework 1 by applying post-hoc explainability methods to a Gradient-Boosted Tree (GBT) classifier — examining why the model assigns high or low recidivism risk to specific defendants, and what those explanations imply for algorithmic governance.

The workflow covers:

1. **Data loading and preprocessing** — Same ProPublica COMPAS dataset and filtering rules as Homework 1.
2. **Model training** — A Logistic Regression (GLM, interpretable by design) and a Gradient-Boosted Tree (black-box) are both fitted following the Lecture 02 code slides.
3. **Group performance metrics** — Per-race accuracy, FPR, FNR, and AUC using the `group_metrics()` function from Lecture 02.
4. **Four key defendants** — Highest-risk and lowest-risk individuals selected from the African-American and Caucasian subgroups.
5. **SHAP analysis** — Global beeswarm summary plot; individual waterfall plots for all four defendants.
6. **LIME analysis** — Local surrogate explanations for all four defendants; bar plots of feature weights.
7. **LIME vs SHAP comparison** — Side-by-side attribution tables; narrative discussion of agreement, divergence, and governance implications.
8. **DiCE counterfactuals** — Three counterfactuals per defendant showing minimal changes to flip the predicted class; immutable features (race, sex) are held fixed and any violation is flagged.
9. **Governance memo** — A ~300-word memo addressed to a hypothetical court auditor summarising findings, method limitations, and monitoring recommendations.

## Python Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical data visualization |
| `scikit-learn` | Preprocessing pipelines, Logistic Regression, GBT, and metrics |
| `shap` | SHAP values — beeswarm and waterfall plots |
| `lime` | LIME local surrogate explanations |
| `dice-ml` | DiCE counterfactual explanations |

## Reproducing the Results

### Prerequisites

- Python 3.9 or later
- `pip` package manager
- Internet connection (the notebook downloads the dataset automatically)

### Steps

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>/HW2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook DNSC-6330-Responsible-ML-homework-02.ipynb
   ```
   Alternatively, open the notebook in JupyterLab, VS Code, or Google Colab and run all cells sequentially (Runtime → Run all).

4. **Data source:** The notebook downloads the dataset automatically from the ProPublica GitHub repository — no manual download is required.

> **Note on runtime:** Training the GBT with 200 estimators and computing SHAP values over the full test set takes approximately 2–4 minutes on a standard laptop CPU. LIME explanations for four individuals add roughly 1–2 minutes.

## Repository Structure

```
HW2/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
└── DNSC-6330-Responsible-ML-homework-02.ipynb  # Main analysis notebook
```

## Data Source

Broward County COMPAS scores and two-year recidivism outcomes from the [ProPublica Machine Bias investigation](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) (2016).

- Dataset: [`compas-scores-two-years.csv`](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv)
- Original R analysis: [ProPublica/compas-analysis](https://github.com/propublica/compas-analysis)

## Key Findings Summary

- `decile_score`, `priors_count`, and `age` are the dominant predictors in both the LR and GBT models.
- The GBT outperforms LR in AUC but exhibits comparable FPR disparity across racial groups, confirming that a more accurate model does not automatically produce fairer outcomes.
- SHAP and LIME converge on the top features for extreme-risk defendants but diverge for mid-range instances where feature correlations distort local approximations.
- DiCE counterfactuals show that low-risk outcomes are achievable through reductions in prior arrests and downgrade of charge degree alone — without requiring changes to immutable characteristics.
