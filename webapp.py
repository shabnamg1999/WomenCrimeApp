import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Crime Against Women Prediction", page_icon="Fevicon.png", layout="centered", initial_sidebar_state="auto", menu_items=None)

def set_bg_hack_url():
    st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://wallpaperboat.com/wp-content/uploads/2019/10/free-website-background-01.jpg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()

st.header("Crime Against Women Prediction Using Machine Learning")

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "Linear Regression"

data_file = st.file_uploader("Upload Crime Data CSV", type=["csv"])

data = None
if data_file is not None:
    data = pd.read_csv(data_file)
    
    
    st.write("### Data Preview")
    st.write(data.head())

if data is not None:
    data.replace("na", np.nan, inplace=True)
    data.dropna(inplace=True)
    data=data.drop(["Id"],axis=1)
    state_column = None
    if 'State/Region' in data.columns:
        state_column = 'State/Region'
    elif 'State' in data.columns:
        state_column = 'State'
    elif 'Region' in data.columns:
        state_column = 'Region'
    
    if state_column:
        selected_state = st.selectbox("Select State/Region", options=data[state_column].unique())
        data = data[data[state_column] == selected_state]
    st.write("### Processed Data")
    st.write(data.head())
    
    year_column = 'Year' if 'Year' in data.columns else None

    column_options = list(data.columns[2:])
    columns = st.multiselect("Select Crime Features", options=column_options, default=column_options[:7])
    target_options = st.multiselect("Select Target Variables (Crimes Against Women)", options=list(data.columns[2:]), default=[list(data.columns)[min(8, len(data.columns) - 7)]])
    
    X = data[columns]
    y = data[target_options]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_names = ["Linear Regression", "Random Forest", "Decision Tree", "Gradient Boosting"]
    selected_model = st.selectbox("Select Model", model_names, index=model_names.index(st.session_state["selected_model"]))
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(max_depth=10, n_estimators=10, random_state=0),
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Gradient Boosting": GradientBoostingRegressor()
    }
    
    model = models[selected_model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write(f"### Model: {selected_model}")
    for i, target in enumerate(target_options):
        st.write(f"**{target} R2 Score:** {r2_score(y_test[target], y_pred[:, i]):.4f}")
        st.write(f"**{target} MSE:** {mean_squared_error(y_test[target], y_pred[:, i]):.4f}")
        st.write(f"**{target} MAE:** {mean_absolute_error(y_test[target], y_pred[:, i]):.4f}")
        st.write(f"**{target} RMSE:** {np.sqrt(mean_squared_error(y_test[target], y_pred[:, i])):.4f}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    if year_column:
        years = data[year_column]
        for target in target_options:
            ax.plot(years, data[target], marker='o', linestyle='-', label=f'Actual {target}')
            ax.plot(years, model.predict(X)[:, target_options.index(target)], marker='s', linestyle='--', label=f'Predicted {target}')
        ax.set_xlabel('Year', fontweight='bold')
    else:
        r = np.arange(len(X))
        for target in target_options:
            ax.plot(r, data[target], marker='o', linestyle='-', label=f'Actual {target}')
            ax.plot(r, model.predict(X)[:, target_options.index(target)], marker='s', linestyle='--', label=f'Predicted {target}')
        ax.set_xlabel('Index', fontweight='bold')
    
    ax.set_ylabel('Number of Cases', fontweight='bold')
    ax.set_title(f'{selected_model} Actual vs Predicted Values')
    ax.legend()
    st.pyplot(fig)
    
    predefined_values = ', '.join(map(str, X.iloc[-1].values))
    st.write("### Predict Crimes for 2025")
    test_input = st.text_input("Enter values (comma-separated)", predefined_values)
    if st.button("Predict 2025 Crime Count"):
        try:
            test_values = list(map(float, test_input.split(',')))
            if len(test_values) != len(columns):
                st.error(f"Please enter exactly {len(columns)} values.")
            else:
                test_df = pd.DataFrame([test_values], columns=columns)
                predictions = model.predict(test_df)
                for i, target in enumerate(target_options):
                    st.write(f"**Predicted Number of {target} Cases in 2025:** {predictions[0, i]:.2f}")
        except ValueError:
            st.error("Invalid input. Please enter numerical values separated by commas.")