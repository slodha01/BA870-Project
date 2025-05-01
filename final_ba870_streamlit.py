# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""Final BA870 Streamlit App"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

st.set_page_config(page_title="Financial Risk Analyzer", layout="centered")

# Load model before conditionals
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

model = load_model()

# Sidebar navigation
tabs = [
    "Project Overview",
    "Dataset Description",
    "Data Dictionary",
    "Models",
    "Upload File",
    "Implications & Next Step"
]
selected_tab = st.sidebar.selectbox("Navigate Project Sections", tabs)

# 1. Project Overview
if selected_tab == "Project Overview":
    st.write("BA870 Project By: Fan Hong (Sally) Kong, Shreya Lodha, Victoria Carlsten")
    st.title("Project Overview")
    st.markdown("<h2 style='color: white;'>Business Problem</h2>", unsafe_allow_html=True)
    st.write("""
    The project aims to detect financial fraud by identifying anomalies in financial statements using the
    **Beneish M-Score** and **Altman’s Z Score**. By applying these fraud detection techniques, we seek to
    flag companies that may be manipulating earnings, reducing financial transparency, and posing risks to
    investors and creditors.
    
    The objective is to enhance fraud detection methodologies for credit risk assessment and financial
    stability evaluation. We will then use the M-score from the Beneish model to try and predict future
    bankruptcy of the companies that engage in manipulation.
    """)
    st.markdown("<h2 style='color: white;'>Data Set</h2>", unsafe_allow_html=True)
    st.write("""
    Our dataset was extracted from WRDS using **Compustat** to compile financial and accounting ratio data
    from **2010–2024** for publicly traded companies in the **technology industry**.
    
    For more information, visit the **‘Dataset Description’** tab.
    """)

# 2. Dataset Description
elif selected_tab == "Dataset Description":
    st.title("Dataset Description")
    st.markdown("""
    **Final Dataset**: Merged_dataset.csv
    *(Includes: Fraud.csv, Additional.csv, Financialdata.csv)*

    All initial datasets were queried and pulled from the **Wharton Research Data Services (WRDS)** via **Compustat**.
    Our team extracted data from the **Fundamentals - Annual** section, filtering for companies in the **technology industry** using SIC codes.

    The **ticker symbols** extracted were then used to pull consistent company data in both **Fraud.csv** and **Additional.csv**.
    The dataset spans from **2010 to 2024** and contains **annual financial reporting** required for calculating the **Beneish M-Score** and **Altman Z-Score**.
    """)
    st.image(
        "https://raw.githubusercontent.com/slodha01/BA870-Project/6d5b485387552e07df58265b64d126ec35eec19e/Photos/Dataset_info.png",
        caption="Merged Dataset: Variable Overview",
        use_container_width=True
    )

# 3. Data Dictionary
elif selected_tab == "Data Dictionary":
    st.title("Dataset Dictionary")

    st.subheader("Fraud.csv Variables")
    fraud_dict = {
        "Column Name": [
            "RESTATEMENT_NOTIFICATION_KEY", "RESTATEMENT_TYPE_FKEY", "RESTATEMENT_TYPE", "FILE_DATE",
            "COMPANY_FKEY", "BEST_EDGAR_TICKER", "is_fraud"
        ],
        "Description": [
            "Restatement Notification Key", "Restatement Type Key", "Restatement Type", "File Date",
            "Company Key", "Best Edgar Ticker", "If company committed fraud (1 = committed, 0 = not commit)"
        ]
    }
    st.dataframe(pd.DataFrame(fraud_dict))

    st.subheader("Financialdata.csv & additional.csv Variables")
    financial_dict = {
        "Column Name": [
            "Ivao", "ivst", "gvkey", "datadate", "fyear", "indfmt", "consol", "popsrc", "datafmt", "tic",
            "conm", "curcd", "fyr", "act", "at", "cogs", "dltt", "dp", "intan", "ist", "lt", "ni", "oancf",
            "ppegt", "rect", "revt", "urect", "xopr", "xsaga", "costat", "gsector", "gsubind", "idbflag", "sic"
        ],
        "Description": [
            "Investment and Advances - Other", "Short-Term Investments - Total", "Global Company Key",
            "Data Date", "Data Year - Fiscal", "Industry Format", "Consolidation Code", "Population Source",
            "Data Format", "Ticker", "Company Name", "ISO Currency Code", "Fiscal Year-end Month",
            "Current Assets - Total", "Assets - Total", "Cost of Goods Sold", "Long-Term Debt - Total",
            "Depreciation and Amortization", "Intangible Assets - Total", "Investment Securities - Total",
            "Liabilities - Total", "Net Income", "Operating Activities - Net Cash Flow",
            "Property, Plant, and Equipment - Total", "Receivables - Total", "Revenue - Total",
            "Receivables (Net)", "Operating Expenses - Total", "SG&A Expense", "Active/Inactive Status",
            "GIC Sector", "GIC Sub-Industries", "International/Domestic Flag", "SIC Code"
        ]
    }
    st.dataframe(pd.DataFrame(financial_dict))

# 4. Models
elif selected_tab == "Models":
    st.title("Beneish M-Score & Altman Z-Score Model")

    st.markdown("### Beneish M-Score Model")
    st.write("""
    We built a **binary classification model** to predict the likelihood of financial fraud using the **Beneish M-Score** framework.
    Models trained:
    - **Random Forest**
    - **XGBoost**

    The target was `is_fraud`, and performance was evaluated using accuracy (93.4%) and a confusion matrix.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "https://raw.githubusercontent.com/slodha01/BA870-Project/eb401490034ef4d92560ed7f27dada0b4ddf18ba/Photos/Radar_Chart.png",
            caption="Radar Chart: Financial Ratios Used",
            use_container_width=True
        )
    with col2:
        st.image(
            "https://raw.githubusercontent.com/slodha01/BA870-Project/ae368f60adf54b40e2990626a083fa6e9c216dd7/Photos/Bar_Chart.png",
            caption="Feature Importance: Random Forest",
            use_container_width=True
        )

    st.markdown("### Altman’s Z-Score")
    st.write("""
    Used to assess bankruptcy risk using five financial ratios. Adjusted for tech sector characteristics.

    **Z-Score Categories**:
    - Z > 2.9: **Safe**
    - 1.23 < Z ≤ 2.9: **Grey Zone**
    - Z < 1.23: **Distress**
    """)

# 5. Upload File
elif selected_tab == "Upload File":
    st.title("Upload Your Financial Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload your file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = {
            'gvkey_x', 'fyear', 'revt', 'cogs', 'rect', 'act', 'at', 'ppegt', 'ivst',
            'dp', 'xsga', 'dltt', 'lt', 'ni', 'oancf'
        }

        if not required_cols.issubset(df.columns):
            st.error(f"Your dataset must include: {', '.join(required_cols)}")
        else:
            df.sort_values(by=['gvkey_x', 'fyear'], inplace=True)
            lag_cols = list(required_cols - {'gvkey_x', 'fyear'})  # exclude identifiers
            for col in lag_cols:
                df[f'{col}_lag'] = df.groupby('gvkey_x')[col].shift(1)

            df['DSRI'] = (df['rect'] / df['revt']) / (df['rect_lag'] / df['revt_lag'])
            df['GMI'] = ((df['revt_lag'] - df['cogs_lag']) / df['revt_lag']) / ((df['revt'] - df['cogs']) / df['revt'])
            df['AQI'] = (1 - (df['act'] + df['ppegt'] + df['ivst']) / df['at']) / \
                        (1 - (df['act_lag'] + df['ppegt_lag'] + df['ivst_lag']) / df['at_lag'])
            df['SGI'] = df['revt'] / df['revt_lag']
            df['DEPI'] = (df['dp_lag'] / (df['ppegt_lag'] + df['dp_lag'])) / (df['dp'] / (df['ppegt'] + df['dp']))
            df['SGAI'] = (df['xsga'] / df['revt']) / (df['xsga_lag'] / df['revt_lag'])
            df['LVGI'] = ((df['lt'] + df['dltt']) / df['at']) / ((df['lt_lag'] + df['dltt_lag']) / df['at_lag'])
            df['TATA'] = (df['ni'] - df['oancf']) / df['at']

            feature_cols = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']
            df_model = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

            is_fraud_pred = model.predict(df_model)
            df.loc[df_model.index, 'rf_is_fraud'] = is_fraud_pred

            df['M_score'] = (
                -4.84 + 0.920 * df['DSRI'] + 0.528 * df['GMI'] + 0.404 * df['AQI'] +
                0.892 * df['SGI'] + 0.115 * df['DEPI'] - 0.172 * df['SGAI'] +
                4.679 * df['TATA'] - 0.327 * df['LVGI']
            )
            df['m_score_pred'] = (df['M_score'] > -1.78).astype(int)

            df['z_score'] = (
                1.2 * (df['act'] - df['lt']) / df['at'] +
                1.4 * df['revt'] / df['at'] +
                3.3 * df['ni'] / df['at'] +
                0.6 * df['revt'] / df['lt'] +
                1.0 * df['revt'] / df['at']
            )

            def classify_z(z):
                if z > 2.9:
                    return 'Safe'
                elif 1.23 < z <= 2.9:
                    return 'Grey Zone'
                else:
                    return 'Distress'
            df['bankruptcy_risk'] = df['z_score'].apply(classify_z)

            st.subheader("Results Summary")
            st.dataframe(df[['rf_is_fraud', 'm_score_pred', 'z_score', 'bankruptcy_risk']].head())

            fraud_summary = df.groupby(['rf_is_fraud', 'bankruptcy_risk']).size().reset_index(name='count')
            st.dataframe(fraud_summary)

            st.subheader("Risk Visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='bankruptcy_risk', hue='rf_is_fraud', palette='Set2', ax=ax)
            ax.set_title('Random Forest Fraud vs. Bankruptcy Risk')
            ax.set_xlabel("Bankruptcy Risk Category")
            ax.set_ylabel("Number of Companies")
            ax.legend(title="Predicted Fraud", labels=["No", "Yes"])
            st.pyplot(fig)
    else:
        st.info("Upload a CSV file to begin.")

# 6. Implications & Next Step
elif selected_tab == "Implications & Next Step":
    st.title("Implications of Fraud Detection")
    st.markdown("### Next Steps")
    st.markdown("""
    - **Refine threshold cutoffs** using industry-specific benchmarks to improve relevance.
    - **Incorporate more variables** (e.g., auditor changes, board shifts, and industry sentiment).
    - **Add NLP modeling** from disclosures to detect qualitative risk cues like vague or evasive language.
    """)
