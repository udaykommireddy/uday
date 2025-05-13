import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App title
st.title("Binary Classification Tool (Flexible Input)")

# Sidebar: CSV upload
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Show a preview of the data
    st.subheader("Uploaded Data Preview:")
    st.dataframe(df.head())

    # Sidebar: select the target column for classification
    target_col = st.sidebar.selectbox("Select the target column", df.columns)

    # Run button
    run = st.sidebar.button("Run")

    if run:
        # Prepare feature matrix X and target vector y
        X = df.drop(columns=[target_col])
        X = X.select_dtypes(include=["number"])
        X = X.fillna(0)
        y = df[target_col]

        # This spinner message will be shown temporarily during execution
        with st.spinner("⏳ Running..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Spinner ends here automatically

        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained! Test Accuracy: {accuracy:.2f}")

        df_with_preds = df.copy()
        df_with_preds["Prediction"] = model.predict(X)
        st.subheader("📈 Full Dataset with Predictions:")
        st.dataframe(df_with_preds)
