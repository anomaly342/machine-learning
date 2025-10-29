import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="AKI Prediction", layout="wide")
# --- 1. Data Definitions ---
all_columns = [
    "Gender",
    "ASAgr",
    "Emer_surg",
    "HT",
    "DM",
    "DLP",
    "COPD",
    "CAD",
    "CVD",
    "NSAIDs",
    "ACEI",
    "ARB",
    "Statin",
    "Diuretics",
    "Dx",
    "Type_Op",
    "Op_app",
    "Side_op",
    "One_lung",
    "Typ_Anal",
    "Hypotension",
    "Hypoxemia",
    "Hypercarbia",
    "offETT",
    "Age",
    "BW",
    "Height",
    "BMI",
    "Dur_anes",
    "Dur_sx",
    "Time_OL",
    "Fluid_ml",
    "Crystalloid_ml",
    "Total_HES_ml",
    "Total_blood_ml",
    "FFP_ml",
    "Bl_loss",
    "Urine",
    "fluid_balance",
    "Ephedrine",
    "Levophed",
    "Hypotension(Mins)",
    "LowestSBP",
    "LowestDBP",
    "Lowest MAP",
    "Pre Hb",
    "Alb",
    "PreCr",
    "PreGFR",
    "NLR1",
]

categorical_cols = [
    "Gender",
    "ASAgr",
    "Emer_surg",
    "HT",
    "DM",
    "DLP",
    "COPD",
    "CAD",
    "CVD",
    "NSAIDs",
    "ACEI",
    "ARB",
    "Statin",
    "Diuretics",
    "Dx",
    "Type_Op",
    "Op_app",
    "Side_op",
    "One_lung",
    "Typ_Anal",
    "Hypotension",
    "Hypoxemia",
    "Hypercarbia",
    "offETT",
]

numerical_cols = [
    "Age",
    "BW",
    "Height",
    "BMI",
    "Dur_anes",
    "Dur_sx",
    "Time_OL",
    "Fluid_ml",
    "Crystalloid_ml",
    "Total_HES_ml",
    "Total_blood_ml",
    "FFP_ml",
    "Bl_loss",
    "Urine",
    "fluid_balance",
    "Ephedrine",
    "Levophed",
    "Hypotension(Mins)",
    "LowestSBP",
    "LowestDBP",
    "Lowest MAP",
    "Pre Hb",
    "Alb",
    "PreCr",
    "PreGFR",
    "NLR1",
]

categorical_allowed = {
    "Gender": [0, 1],
    "ASAgr": [0, 1, 2, 3],
    "Emer_surg": [0, 1],
    "HT": [0, 1],
    "DM": [0, 1],
    "DLP": [0, 1],
    "COPD": [0, 1],
    "CAD": [0, 1],
    "CVD": [0, 1],
    "NSAIDs": [0, 1],
    "ACEI": [0, 1],
    "ARB": [0, 1],
    "Statin": [0, 1],
    "Diuretics": [0, 1],
    "Dx": [0, 1, 2],
    "Type_Op": [0, 1, 2, 3, 4, 5],
    "Op_app": [0, 1],
    "Side_op": [0, 1, 2],
    "One_lung": [0, 1],
    "Typ_Anal": [0, 1],
    "Hypotension": [0, 1],
    "Hypoxemia": [0, 1],
    "Hypercarbia": [0, 1],
    "offETT": [0, 1, 2, 3],
}

code_hints = {
    "Gender": {0: "Female", 1: "Male"},
    "ASAgr": {0: "I", 1: "II", 2: "III", 3: "IV+"},
    "Dx": {0: "Benign", 1: "Early Cancer", 2: "Advanced Cancer"},
    "Side_op": {0: "Left", 1: "Right", 2: "Bilateral"},
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
    "One_lung": {0: "No", 1: "Yes"},
    "Typ_Anal": {0: "No", 1: "Yes"},
    "Hypotension": {0: "No", 1: "Yes"},
    "Hypoxemia": {0: "No", 1: "Yes"},
    "Hypercarbia": {0: "No", 1: "Yes"},
    "Op_app": {0: "Open", 1: "Minimal Access"},
    "Type_Op": {i: f"Type {i}" for i in range(6)},
    "offETT": {0: "None", 1: "Code 1", 2: "Code 2", 3: "Code 3"},
}

clinical_weights = {
    "Lowest MAP": 1.8,  # ↓ มาก = เสี่ยงสูง
    "Hypotension(Mins)": 1.8,  # นาน = เสี่ยงสูง
    "LowestSBP": 1.3,
    "LowestDBP": 1.3,
    "Levophed": 1.7,  # ใช้เยอะ = เคสหนัก
    "Ephedrine": 1.15,
    "Bl_loss": 1.6,
    "Total_blood_ml": 1.6,
    "FFP_ml": 1.6,
    "Alb": 1.9,  # Alb ต่ำ = เสี่ยงชัด (ดันให้โมเดล “แคร์” มาก)
    "PreCr": 1.6,  # สูง = เสี่ยง
    "PreGFR": 1.4,  # ต่ำ = เสี่ยง
    "Urine": 1.4,  # ต่ำ = เสี่ยง
    "Pre Hb": 1.15,  # โลหิตจาง = เสี่ยง
    "NLR1": 1.15,  # inflammation marker
    "Age": 1.15,
}

default_values = {
    "Age": 39,
    "Gender": 0,
    "ASAgr": 1,
    "Emer_surg": 0,
    "BW": 38,
    "Height": 155,
    "BMI": 15.81686,
    "HT": 0,
    "DM": 0,
    "DLP": 0,
    "COPD": 0,
    "CAD": 0,
    "CVD": 0,
    "NSAIDs": 0,
    "ACEI": 0,
    "ARB": 0,
    "Statin": 0,
    "Diuretics": 0,
    "Dx": 1,
    "Type_Op": 3,
    "Op_app": 1,
    "Side_op": 1,
    "One_lung": 1,
    "Typ_Anal": 0,
    "Hypotension": 0,
    "Hypoxemia": 0,
    "Hypercarbia": 1,
    "offETT": 2,
    "Dur_anes": 255,
    "Dur_sx": 195,
    "Time_OL": 170,
    "Fluid_ml": 5027,
    "Crystalloid_ml": 1950,
    "Total_HES_ml": 1300,
    "Total_blood_ml": 830,
    "FFP_ml": 947,
    "Bl_loss": 3000,
    "Urine": 195,
    "fluid_balance": 4002,
    "Ephedrine": 12,
    "Levophed": 315,
    "Hypotension(Mins)": 10,
    "LowestSBP": 78,
    "LowestDBP": 42,
    "Lowest MAP": 54,
    "Pre Hb": 10.8,
    "Alb": 2.2,
    "PreCr": 0.7,
    "PreGFR": 105.17,
    "NLR1": 89,
}


def apply_feature_weights(X, feature_names):
    if isinstance(X, pd.DataFrame):
        Xw = X.copy()
        for col in Xw.columns:
            w = clinical_weights.get(col, 1.0)

            Xw[col] = Xw[col] * w
        return Xw
    else:
        Xw = np.array(X, dtype=float, copy=True)
        for i, c in enumerate(feature_names):
            Xw[:, i] = Xw[:, i] * clinical_weights.get(c, 1.0)
        return Xw


preprocess_weighted = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            [c for c in all_columns if c in categorical_cols],
        ),
        (
            "num",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "weight_num",
                        FunctionTransformer(
                            func=apply_feature_weights,
                            kw_args={
                                "feature_names": [
                                    c for c in all_columns if c in numerical_cols
                                ]
                            },
                        ),
                    ),
                ]
            ),
            [c for c in all_columns if c in numerical_cols],
        ),
    ],
    remainder="drop",
)


categorical_columns = list(categorical_allowed.keys())
numerical_columns = [col for col in all_columns if col not in categorical_columns]

# --- 2. Load XGB JSON Model ---
MODEL_PATH = "xgb_model.json"
xgb_model = XGBClassifier()
if os.path.exists(MODEL_PATH):
    xgb_model.load_model(MODEL_PATH)
    st.success("XGBoost model loaded successfully!")
else:
    st.warning(f"Model file '{MODEL_PATH}' not found. Predictions disabled.")


# --- 3. Prediction Function ---
def predict_icu_outcome(input_df):
    proba = xgb_model.predict_proba(input_df)[0]
    pred_class = int(xgb_model.predict(input_df)[0])
    risk_level_map = {
        0: "Safe",
        1: "Stage 1",
        2: "Stage 2",
        3: "Stage 3 (Critical Concern)",
    }
    return {
        "Predicted Class Code": pred_class,
        "Risk Level": risk_level_map.get(pred_class, "Unknown"),
        "Raw Probabilities": {i: f"{p*100:.2f}%" for i, p in enumerate(proba)},
    }


# --- 4. Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="Medical Data Input")
st.title("Patient and Procedure Data Collection Form")
st.markdown(
    "Please enter the required information for the patient and surgical procedure below."
)

total_columns = len(all_columns)
st.info(f"The form contains **{total_columns}** unique data points (columns).")

input_data = {}


def get_input_widget(col_name, columns, col_index):
    current_col = columns[col_index]

    # Use your real defaults if available
    default_val = default_values.get(col_name, 0)

    if col_name in categorical_columns:
        options = categorical_allowed[col_name]
        radio_options = [
            f"{opt} ({code_hints.get(col_name, {}).get(opt, 'No info')})"
            for opt in options
        ]

        # find which option matches the default value
        try:
            default_index = options.index(int(default_val))
        except (ValueError, TypeError):
            default_index = 0

        selected_option = current_col.radio(
            f"**{col_name.replace('_', ' ')}**",
            radio_options,
            index=default_index,
        )
        input_data[col_name] = int(selected_option.split(" ")[0])

    elif col_name in numerical_columns:
        input_data[col_name] = current_col.text_input(
            f"**{col_name.replace('_', ' ')}**",
            value=str(default_val),
        )


def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i : i + size]


# --- 5. UI Sections ---

# 5.1 Patient & Baseline
st.header("Patient & Baseline Characteristics")
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    cat_baseline = [
        "Gender",
        "ASAgr",
        "Emer_surg",
        "HT",
        "DM",
        "DLP",
        "COPD",
        "CAD",
        "CVD",
    ]
    cols = [col1, col2, col3]
    for i, col_name in enumerate(cat_baseline):
        get_input_widget(col_name, cols, i % 3)
    st.markdown("---")
    num_baseline = ["Age", "BW", "Height", "BMI"]
    col4, col5, col6, col7 = st.columns(4)
    cols_num = [col4, col5, col6, col7]
    for i, col_name in enumerate(num_baseline):
        get_input_widget(col_name, cols_num, i % 4)

# 5.2 Medication & Dx
with st.expander("2. Medications & Diagnosis"):
    col1, col2, col3 = st.columns(3)
    med_dx = ["NSAIDs", "ACEI", "ARB", "Statin", "Diuretics", "Dx"]
    cols = [col1, col2, col3]
    for i, col_name in enumerate(med_dx):
        get_input_widget(col_name, cols, i % 3)

# 5.3 Surgical & Anesthesia
with st.expander("3. Surgical & Anesthesia Details"):
    col1, col2, col3 = st.columns(3)
    surg_details = ["Type_Op", "Op_app", "Side_op", "One_lung", "Typ_Anal"]
    cols = [col1, col2, col3]
    for i, col_name in enumerate(surg_details):
        get_input_widget(col_name, cols, i % 3)
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    durations = ["Dur_anes", "Dur_sx", "Time_OL"]
    cols_num = [col4, col5, col6]
    for i, col_name in enumerate(durations):
        get_input_widget(col_name, cols_num, i % 3)

# 5.4 Hemodynamics & Events
with st.expander("4. Hemodynamics & Events"):
    col1, col2, col3 = st.columns(3)
    events = ["Hypotension", "Hypoxemia", "Hypercarbia", "offETT"]
    cols = [col1, col2, col3]
    for i, col_name in enumerate(events):
        get_input_widget(col_name, cols, i % 3)
    st.markdown("---")
    col4, col5, col6, col7 = st.columns(4)
    hemo_num = ["Hypotension(Mins)", "LowestSBP", "LowestDBP", "Lowest MAP"]
    cols_num = [col4, col5, col6, col7]
    for i, col_name in enumerate(hemo_num):
        get_input_widget(col_name, cols_num, i % 4)
    st.markdown("---")
    col8, col9 = st.columns(2)
    vaso_drugs = ["Ephedrine", "Levophed"]
    cols_num2 = [col8, col9]
    for i, col_name in enumerate(vaso_drugs):
        get_input_widget(col_name, cols_num2, i % 2)

# 5.5 Fluid & Output
with st.expander("5. Fluid, Blood, and Output"):
    for chunk in chunk_list(
        [
            "Fluid_ml",
            "Crystalloid_ml",
            "Total_HES_ml",
            "Total_blood_ml",
            "FFP_ml",
            "Bl_loss",
        ],
        3,
    ):
        cols = st.columns(3)
        for i, col_name in enumerate(chunk):
            get_input_widget(col_name, cols, i)
    col7, col8, col9 = st.columns(3)
    for i, col_name in enumerate(["Urine", "fluid_balance"]):
        get_input_widget(col_name, [col7, col8, col9], i % 3)

# 5.6 Labs
with st.expander("6. Pre-operative Labs & Ratios"):
    cols = st.columns(5)
    for i, col_name in enumerate(["Pre Hb", "Alb", "PreCr", "PreGFR", "NLR1"]):
        get_input_widget(col_name, cols, i % 5)

# --- 6. Submit & Predict ---
st.markdown("---")
if st.button("Submit Data and Predict", type="primary"):
    validation_error = False
    try:
        validated_data = {}
        for col, val in input_data.items():
            if col in categorical_columns:
                validated_data[col] = int(val)
            else:
                validated_data[col] = float(val)
    except ValueError as e:
        st.error(f"Validation Error: {e}")
        validation_error = True

    if not validation_error:
        TARGET = "AKI"
        THRESH = 0.5
        RANDOM_STATE = 42
        df = pd.read_csv("AKI.csv", dtype=str)
        df.columns = [c.strip() for c in df.columns]

        df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
        df = df.apply(
            lambda s: s.str.replace(",", "", regex=False) if s.dtype == "object" else s
        )
        df = df.replace({r"^\s*\.\s*$": np.nan, r"^\s*$": np.nan}, regex=True)

        def to_numeric_clean(s: pd.Series) -> pd.Series:
            x = s.astype(str).str.strip().str.replace(",", "", regex=False)
            x = x.replace(
                {"": np.nan, ".": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan}
            )
            return pd.to_numeric(x, errors="coerce")

        # แปลง numerical cols
        for col in [c for c in numerical_cols if c in df.columns]:
            df[col] = to_numeric_clean(df[col])

        # แปลง categorical cols เป็นตัวเลข (รหัส)
        for col in [c for c in categorical_cols if c in df.columns]:
            df[col] = to_numeric_clean(df[col])

        df[TARGET] = to_numeric_clean(df[TARGET])
        rows_before = len(df)
        df = df[~df[TARGET].isna()].copy()
        df[TARGET] = df[TARGET].astype(int)

        invalid_rows = pd.Series(False, index=df.index)

        for col, allowed in categorical_allowed.items():
            if col not in df.columns:
                continue
            mask_invalid = df[col].notna() & ~df[col].isin(allowed)
            if mask_invalid.any():

                invalid_rows |= mask_invalid

        if invalid_rows.any():
            rows_before = len(df)
            df = df[~invalid_rows].copy()

        num_in = [c for c in numerical_cols if c in df.columns]
        miss_rate = df[num_in].isna().mean()
        low_miss = miss_rate[(miss_rate > 0) & (miss_rate < THRESH)].index.tolist()
        high_miss = miss_rate[miss_rate >= THRESH].index.tolist()

        if low_miss:
            med = df[low_miss].median(numeric_only=True)
            df[low_miss] = df[low_miss].fillna(med)

        remain = [c for c in num_in if df[c].isna().any()]
        if remain:
            mice = IterativeImputer(
                random_state=RANDOM_STATE, max_iter=15, initial_strategy="median"
            )
            X = df[num_in].copy()
            Xi = pd.DataFrame(mice.fit_transform(X), columns=num_in, index=df.index)
            df[remain] = Xi[remain]

        cat_in = [c for c in categorical_cols if c in df.columns and c != TARGET]
        if cat_in:
            mode_imp = SimpleImputer(strategy="most_frequent")
            df[cat_in] = mode_imp.fit_transform(df[cat_in])

        X = df.drop(columns=["Date", "AKI"])
        y = df["AKI"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        st.success("Input data successfully captured and validated!")
        input_df = pd.DataFrame([validated_data], columns=all_columns)
        preprocess_weighted.fit(X_train)
        processed_input = preprocess_weighted.transform(input_df)

        if "xgb_model" in globals():
            try:
                prediction_details = predict_icu_outcome(processed_input)
                st.subheader(f"ICU Risk Prediction: {prediction_details['Risk Level']}")
                st.json(prediction_details)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
                st.error(f"Prediction failed: {e}")
