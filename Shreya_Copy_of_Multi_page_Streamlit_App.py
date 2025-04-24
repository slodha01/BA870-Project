

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Financial Risk Analyzer", layout="centered")

# Sidebar navigation
tabs = [
    "Project Overview",
    "Dataset Description",
    "Data Dictionary",
    "Upload File",
    "Beneish Model",
    "Implications - Detecting Fraud",
    "Next Step"
]
selected_tab = st.sidebar.selectbox("Navigate Project Sections", tabs)

# 1. Project Overview
if selected_tab == "Project Overview":
    st.title("Project Overview")
    st.markdown("<h2 style='color: navy;'>Business Problem</h2>", unsafe_allow_html=True)
    st.write("""

    By: Fan Hong (Sally) Kong, Shreya Lodha, Victoria Carlsten

    The project aims to detect financial fraud by identifying anomalies in financial statements using the
    **Beneish M-Score** and **Altman’s Z Score**. By applying these fraud detection techniques, we seek to
    flag companies that may be manipulating earnings, reducing financial transparency, and posing risks to
    investors and creditors.

    The objective is to enhance fraud detection methodologies for credit risk assessment and financial
    stability evaluation. We will then use the M-score from the Beneish model to try and predict future
    bankruptcy of the companies that engage in manipulation.
    """)
    st.markdown("<h2 style='color: navy;'>Data Set</h2>", unsafe_allow_html=True)
    st.write("""
    Our dataset was extracted from WRDS using **Compustat** to compile financial and accounting ratio data
    from **2010–2025** for publicly traded companies in the **technology industry**.

    For more information, visit the **‘Dataset Description’** tab.
    """)

# 2. Dataset Description
elif selected_tab == "Dataset Description":
    st.title("Dataset Description")
    st.write("_____")

# 3. Data Dictionary
elif selected_tab == "Data Dictionary":
    st.title("Dataset Dictionary")

    # Fraud.csv Data Dictionary
    st.subheader("Fraud.csv Variables")
    fraud_dict = {
        "Column Name": [
            "RESTATEMENT_NOTIFICATION_KEY",
            "RESTATEMENT_TYPE_FKEY",
            "RESTATEMENT_TYPE",
            "FILE_DATE",
            "COMPANY_FKEY",
            "BEST_EDGAR_TICKER",
            "is_fraud"
        ],
        "Description": [
            "Restatement Notification Key",
            "Restatement Type Key",
            "Restatement Type",
            "File Date",
            "Company Key",
            "Best Edgar Ticker",
            "If company committed fraud (1 = committed, 0 = not commit)"
        ]
    }
    st.dataframe(pd.DataFrame(fraud_dict))

    # Financialdata.csv & additional.csv Data Dictionary
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
            "Current Assets - Total", "Current Assets - Total", "Cost of Goods Sold", "Long-Term Debt - Total",
            "Depreciation and Amortization", "Intangible Assets - Total", "Investment Securities - Total", "Liabilities - Total", "Net Income", "Operating Activities - Net Cash Flow",
            "Property, Plant, and Equipment - Total", "Receivables - Total", "Revenue - Total", "Receivables (Net)",
            "Operating Expenses - Total", "Selling, General, & Administrative Expense", "Active/ Inactive Status Market",
            "GIC Sector", "GIC Sub-Industries", "International, Domestic, Both Indicator",
            "Standard Industry Classification Code"
        ]
    }
    st.dataframe(pd.DataFrame(financial_dict))

# 4. Upload File
elif selected_tab == "Upload File":
    st.title("Upload Your Processed Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload your file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        required_cols = {'is_fraud', 'z_score', 'bankruptcy_risk'}
        if not required_cols.issubset(df.columns):
            st.error(f"Your dataset must contain the following columns: {', '.join(required_cols)}")
        else:
            st.success("File successfully loaded!")

            with st.expander("View raw data"):
                st.dataframe(df.head())

            st.subheader("Fraud vs. Bankruptcy Risk Counts")
            summary = df.groupby(['is_fraud', 'bankruptcy_risk']).size().reset_index(name='count')
            st.dataframe(summary)

            st.subheader("Risk Distribution Chart")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='bankruptcy_risk', hue='is_fraud', palette='Set2', ax=ax)
            ax.set_title('Fraud Risk vs. Bankruptcy Risk')
            ax.set_xlabel("Bankruptcy Risk Category")
            ax.set_ylabel("Number of Companies")
            ax.legend(title="Fraudulent (Beneish M-Score)", labels=["No", "Yes"])
            st.pyplot(fig)

            st.subheader("Altman Z′-Score Summary")
            stats = df.groupby('is_fraud')['z_score'].agg(['mean', 'median', 'std', 'count']).rename(
                index={0: 'Non-Fraud', 1: 'Fraud'}
            )
            st.dataframe(stats)
    else:
        st.info("Upload a CSV to begin.")

# 5. Beneish Model
elif selected_tab == "Beneish Model":
    st.title("Beneish M-Score Model")
    st.write("Insert Beneish M-Score analysis and logic here.")
    # st.markdown("## 🧠 Beneish M-Score Model Overview")
    # st.markdown("""
    # The Beneish M-Score is used to detect earnings manipulation by analyzing 8 financial ratios.
    # In this project, we trained a machine learning model on WRDS Compustat data to classify companies 
    # as potential manipulators (M-Score > -2.22).

    # We used the following ratios as input features:
    # - DSRI: Days Sales in Receivables Index
    # - GMI: Gross Margin Index
    # - AQI: Asset Quality Index
    # - SGI: Sales Growth Index
    # - DEPI: Depreciation Index
    # - SGAI: Sales, General, and Admin Expenses Index
    # - LVGI: Leverage Index
    # - TATA: Total Accruals to Total Assets

    # We trained a Random Forest Regressor to identify manipulators. We also used the XGBoost, but it didn't perform as well.
    # """)
    # st.markdown("### 📄 Model Code (Simplified)")

    # code = '''
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.model_selection import train_test_split

    # # Features and target
    # X = fraud_model_data[["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]]
    # y = fraud_model_data["is_fraud"]

    # # Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # # Model
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)
    # '''

    # st.code(code, language='python')

    # st.markdown("### ✅ Model Performance")

    # st.markdown("- **Accuracy**: 88.6%")
    # st.markdown("- **Precision**: 82.3%")
    # st.markdown("- **Recall**: 76.5%")

# 6. Implications - Detecting Fraud
elif selected_tab == "Implications - Detecting Fraud":
    st.title("Implications of Fraud Detection")
    st.write("Insert interpretation of fraud signals and business implications here.")

# 7. Next Step
elif selected_tab == "Next Step":
    st.title("Next Steps")
    st.write("Insert recommendations, improvements, and roadmap.")
