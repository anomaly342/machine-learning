import streamlit as st

st.set_page_config(page_title="AKI Prediction Overview", layout="wide", page_icon="🩺")

# --- Header ---
st.title("🩺 Postoperative AKI Prediction Project")
st.markdown(
    """
Acute Kidney Injury (AKI) is a serious postoperative complication associated with increased
morbidity, prolonged hospital stays, and higher mortality rates. Traditional risk assessment
methods rely on static clinical scores and limited preoperative indicators, which may not fully
capture the complex interplay of perioperative factors.

This project leverages **machine learning** techniques to predict postoperative AKI in patients
undergoing thoracic surgery, enabling early intervention and improved patient outcomes.
"""
)

# --- Dataset Info ---
st.header("Dataset Overview")
st.markdown(
    """
- **Total rows:** 1,947 (before data cleansing)
- **Total columns:** 50
- **Dropped columns:** `Date`, `AKI` (target)
- **Feature classes:**
  - **Categorical:** Gender, ASAgr, Emer_surg, HT, DM, DLP, COPD, CAD, CVD, NSAIDs, ACEI, ARB, Statin, Diuretics, Dx, Type_Op, Op_app, Side_op, One_lung, Typ_Anal, Hypotension, Hypoxemia, Hypercarbia, offETT
  - **Numerical:** Age, BW, Height, BMI, Dur_anes, Dur_sx, Time_OL, Fluid_ml, Crystalloid_ml, Total_HES_ml, Total_blood_ml, FFP_ml, Bl_loss, Urine, fluid_balance, Ephedrine, Levophed, Hypotension(Mins), LowestSBP, LowestDBP, Lowest MAP, Pre Hb, Alb, PreCr, PreGFR, NLR1
"""
)

# --- Categorical Feature Encoding ---
with st.expander("Categorical Feature Encoding (click to expand)"):
    categorical_encoding = {
        "Gender": {0: "Female", 1: "Male"},
        "ASAgr": {0: "I", 1: "II", 2: "III", 3: "IV+"},
        "Emer_surg": {0: "No", 1: "Yes"},
        "HT": {0: "No", 1: "Yes"},
        "DM": {0: "No", 1: "Yes"},
        "DLP": {0: "No", 1: "Yes"},
        "COPD": {0: "No", 1: "Yes"},
        "CAD": {0: "No", 1: "Yes"},
        "CVD": {0: "No", 1: "Yes"},
        "NSAIDs": {0: "No", 1: "Yes"},
        "ACEI": {0: "No", 1: "Yes"},
        "ARB": {0: "No", 1: "Yes"},
        "Statin": {0: "No", 1: "Yes"},
        "Diuretics": {0: "No", 1: "Yes"},
        "Dx": {0: "Benign", 1: "Early Cancer", 2: "Advanced Cancer"},
        "Type_Op": {
            0: "Type 0",
            1: "Type 1",
            2: "Type 2",
            3: "Type 3",
            4: "Type 4",
            5: "Type 5",
        },
        "Op_app": {0: "Open", 1: "Minimal Access"},
        "Side_op": {0: "Left", 1: "Right", 2: "Bilateral"},
        "One_lung": {0: "No", 1: "Yes"},
        "Typ_Anal": {0: "No", 1: "Yes"},
        "Hypotension": {0: "No", 1: "Yes"},
        "Hypoxemia": {0: "No", 1: "Yes"},
        "Hypercarbia": {0: "No", 1: "Yes"},
        "offETT": {0: "None", 1: "Code 1", 2: "Code 2", 3: "Code 3"},
    }

    for col, mapping in categorical_encoding.items():
        st.subheader(col)
        for key, val in mapping.items():
            st.write(f"{key} = {val}")

# --- Numerical Features & Clinical Weights ---
with st.expander("Numerical Features & Clinical Weights (click to expand)"):
    numerical_weights = {
        "Age": 1.15,
        "BW": 1.0,
        "Height": 1.0,
        "BMI": 1.0,
        "Dur_anes": 1.0,
        "Dur_sx": 1.0,
        "Time_OL": 1.0,
        "Fluid_ml": 1.0,
        "Crystalloid_ml": 1.0,
        "Total_HES_ml": 1.0,
        "Total_blood_ml": 1.6,
        "FFP_ml": 1.6,
        "Bl_loss": 1.6,
        "Urine": 1.4,
        "fluid_balance": 1.0,
        "Ephedrine": 1.15,
        "Levophed": 1.7,
        "Hypotension(Mins)": 1.8,
        "LowestSBP": 1.3,
        "LowestDBP": 1.3,
        "Lowest MAP": 1.8,
        "Pre Hb": 1.15,
        "Alb": 1.9,
        "PreCr": 1.6,
        "PreGFR": 1.4,
        "NLR1": 1.15,
    }

    for feature, weight in numerical_weights.items():
        st.write(f"**{feature}** — Weight: {weight}")
# --- Model & Prediction Info ---
st.header("Machine Learning Model")
st.markdown(
    """
- **Model Used:** XGBoost Classifier  
- **Input:** Patient and perioperative clinical data  
- **Output:** AKI Risk Level:
  - Safe
  - Stage 1
  - Stage 2
  - Stage 3 (Critical Concern)
- **Preprocessing:** Weighted numerical features + one-hot encoding categorical features  
- **Validation:** Train-test split with stratification, iterative imputer for missing values
"""
)

st.subheader("Key Benefits")
st.markdown(
    """
- Predicts AKI **before symptoms manifest**, enabling early intervention  
- Accounts for **complex interactions** among perioperative factors  
- Can be integrated with hospital EHRs or surgical monitoring systems  
- Adaptable for research, clinical decision support, and risk stratification
"""
)
with st.expander("Data Preprocessing Steps (click to expand)"):
    st.markdown(
        """
### 1️⃣ Handling Invalid Categorical Values
Some categorical columns may contain **values that are not allowed** or are missing (`NaN`).  
- **Invalid values** are any values **not listed in the allowed set** for that column.
- **Solution:** Replace them with the **mode** (most frequent value) of that column.

**Why:**  
- Categorical features cannot use mean or median.  
- Mode ensures the value is realistic and representative of the dataset.

**Example:**

---

### 2️⃣ Removing Rows with Missing Target ("AKI")
- Any row where the **AKI column is blank or NaN** cannot be used for training or evaluation.  
- **Solution:** Drop those rows entirely from the dataset.

**Why:**  
- Target labels are required for supervised machine learning.  
- Missing targets would break training and produce invalid predictions.

---

### 3️⃣ Handling Missing Numerical Values
- Some numerical columns may have **NaN or missing values**.  
- **Solution:** Replace NaN with the **mean** of the column.

**Why:**  
- Using the mean preserves the overall distribution.  
- Prevents data loss while maintaining the statistical properties of the dataset.

**Example:**
"""
    )
