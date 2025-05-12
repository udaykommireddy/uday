import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

st.title("🧠 Binary Classification Tool (Flexible Input)")

st.sidebar.header("📁 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        st.write(df.head())

        # Ask for target column
        target_column = st.sidebar.selectbox("Select the target column", df.columns)

        if target_column not in df.columns:
            st.error(f"❌ The selected target column '{target_column}' is not found in the dataset.")
        else:
            # Separate target and features
            y = df[target_column]
            X = df.drop(columns=[target_column])

            # Keep only numeric columns
            X_numeric = X.select_dtypes(include=['number'])

            # Impute missing values
            imputer = SimpleImputer(strategy="mean")
            X_filled = imputer.fit_transform(X_numeric)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filled)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Train logistic regression on this data
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Predict and show accuracy
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"✅ Model trained! Test Accuracy: {acc:.2f}")

            # Predict on entire dataset
            full_preds = model.predict(X_scaled)
            df['Prediction'] = full_preds
            st.write("📈 Full Dataset with Predictions:")
            st.write(df)

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
else:
    st.info("👈 Upload a CSV file with a target column to start.")
