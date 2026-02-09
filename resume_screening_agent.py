# ============================================================
# 📌 PROJECT: RESUME SCREENING AGENT
# USING NLP + MACHINE LEARNING + AGENTIC AI LOGIC
#
# ROLE ALIGNMENT:
# - Data Science Trainer
# - ML Trainer
# - GenAI / Agentic AI Trainer
#
# ============================================================


# ============================================================
# 1️⃣ IMPORT REQUIRED LIBRARIES
# ============================================================

# NumPy is used for numerical operations
import numpy as np

# CountVectorizer converts text into numbers
from sklearn.feature_extraction.text import CountVectorizer

# Logistic Regression is used for binary classification
from sklearn.linear_model import LogisticRegression


# ============================================================
# 2️⃣ SYNTHETIC DATASET (SAFE FOR INTERVIEW)
# ============================================================
# These are SAMPLE resumes written as plain text
# (In real enterprise systems, resumes come from PDFs, portals, etc.)

resumes = [
    "Python ML SQL experience",              # Strong DS resume
    "Java backend developer",                # Non-DS resume
    "Python Data Science ML NLP",             # Strong DS resume
    "Sales marketing executive",              # Non-technical
    "Python SQL Statistics"                   # DS fundamentals
]

# Labels:
# 1 → Shortlisted
# 0 → Rejected
labels = [1, 0, 1, 0, 1]


# ============================================================
# 3️⃣ NLP STEP — TEXT TO NUMBERS
# ============================================================
# ML models cannot understand text
# So we convert resume text into numeric features

vectorizer = CountVectorizer()     # Bag-of-Words model

# Fit and transform resumes into numeric matrix
X = vectorizer.fit_transform(resumes)


# ============================================================
# 4️⃣ MACHINE LEARNING MODEL — LOGISTIC REGRESSION
# ============================================================
# Logistic Regression outputs PROBABILITY
# Probability helps in decision making (shortlist / reject / escalate)

model = LogisticRegression()

# Train model on synthetic dataset
model.fit(X, labels)


# ============================================================
# 5️⃣ AGENTIC AI DECISION FUNCTION
# ============================================================
def resume_screening_agent(resume_text):
    """
    PURPOSE:
    This function acts like an AI AGENT.
    
    PIPELINE:
    Resume Text → NLP → ML Score → Decision
    
    RETURNS:
    Resume score and decision
    """

    # Convert incoming resume text to numeric features
    resume_vector = vectorizer.transform([resume_text])

    # Predict probability of being shortlisted
    probability = model.predict_proba(resume_vector)[0][1]

    # Agentic decision logic (threshold-based)
    if probability >= 0.70:
        decision = "SHORTLIST"
    elif probability >= 0.40:
        decision = "ESCALATE TO HUMAN"
    else:
        decision = "REJECT"

    # Return result as dictionary (easy to explain in interview)
    return {
        "Resume Text": resume_text,
        "Shortlist Probability": round(probability, 2),
        "Final Decision": decision
    }


# ============================================================
# 6️⃣ RUN THE AGENT (TEST CASE)
# ============================================================

test_resume = "Python ML Statistics SQL"

result = resume_screening_agent(test_resume)

print("Resume Screening Result:")
print(result)


# ============================================================
# 7️⃣ INTERVIEW EXPLANATION (MENTALLY REMEMBER)
# ============================================================
#
# - Resume is converted to numbers using NLP
# - ML model calculates probability
# - Agent decides next action
# - Escalation ensures responsible AI
#
# ============================================================
# 8️⃣ WHY THIS IS AGENTIC AI (INTERVIEW ANSWER)
# ============================================================
#
# ❌ Traditional ML:
# - Only predicts Yes/No
#
# ✅ Agentic AI:
# - Predicts
# - Decides next step
# - Can escalate to human
#
# ============================================================
# 9️⃣ ENTERPRISE USE CASES
# ============================================================
#
# - Resume screening automation
# - Campus hiring
# - Internal role matching
# - Vendor profile screening
#
# ============================================================
# 🔚 END OF PROJECT
# ============================================================
