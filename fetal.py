# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

#AI used to help get background color syntax
def color_prediction(val):
    if val == 'Normal':
        color = 'lime'
    elif val == 'Suspect':
        color = 'yellow'
    elif val == 'Pathological':
        color = 'orange'
    else:
        color = 'white'
    return f'background-color: {color}; color: black'

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 
st.image('fetal_health_image.gif', width = 400)
st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")
st.sidebar.subheader('**Fetal Health Features Input**')
fetal_file = st.sidebar.file_uploader("Upload your data",help="File must be in CSV format", type=['csv'])
st.sidebar.warning("⚠️ Ensure your data strictly follows the format outlined below.")
st.sidebar.write(pd.read_csv('fetal_health_user.csv').head(5))
model_choice = st.sidebar.radio('Choose Model for Prediction', options = ['Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting'])
st.sidebar.info(f"✓ You selected: {model_choice}")


if fetal_file is None:
    st.info("**Please upload data to proceed.**")
else:
    st.header(f"**Predicting Fetal Health Class Using {model_choice} Model**")
    if model_choice == 'Decision Tree':
        dt_pickle = open('decision_tree_fetal.pickle', 'rb')
        clf = pickle.load(dt_pickle)
        dt_pickle.close()
        user_df = pd.read_csv(fetal_file)
        
        user_pred = clf.predict(user_df)
        prob = clf.predict_proba(user_df)

        mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
        user_pred_fetal = [mapping.get(pred, 'Unknown') for pred in user_pred]

        user_prob_fetal = prob.max(axis=1) * 100
        user_df['Predicted Fetal Health'] = user_pred_fetal
        user_df['Prediction Probability'] = np.round(user_prob_fetal, 1).astype(str) + '%'
        styled_df = user_df.style.applymap(color_prediction, subset=['Predicted Fetal Health'])
        st.dataframe(styled_df)

        st.header("**Model Performance and Insights**")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report","Feature Importance"])

        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_dt.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_dt.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition.")
        
        # Tab 3: Feature Importance Visualization
        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_dt.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

    elif model_choice == 'Random Forest':
        rf_pickle = open('random_forest_fetal.pickle', 'rb')
        clf2 = pickle.load(rf_pickle)
        rf_pickle.close()
        user_df = pd.read_csv(fetal_file)
        
        user_pred = clf2.predict(user_df)
        prob = clf2.predict_proba(user_df)

        mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
        user_pred_fetal = [mapping.get(pred, 'Unknown') for pred in user_pred]

        user_prob_fetal = prob.max(axis=1) * 100
        user_df['Predicted Fetal Health'] = user_pred_fetal
        user_df['Prediction Probability'] = np.round(user_prob_fetal, 1).astype(str) + '%'
        styled_df = user_df.style.applymap(color_prediction, subset=['Predicted Fetal Health'])
        st.dataframe(styled_df)

        st.header("**Model Performance and Insights**")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report","Feature Importance"])

        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_rf.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_rf.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition.")
        
        # Tab 3: Feature Importance Visualization
        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_rf.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

    elif model_choice == 'AdaBoost':
        ada_pickle = open('adaboost_fetal.pickle', 'rb')
        clf3 = pickle.load(ada_pickle)
        ada_pickle.close()
        user_df = pd.read_csv(fetal_file)
        
        user_pred = clf3.predict(user_df)
        prob = clf3.predict_proba(user_df)

        mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
        user_pred_fetal = [mapping.get(pred, 'Unknown') for pred in user_pred]

        user_prob_fetal = prob.max(axis=1) * 100
        user_df['Predicted Fetal Health'] = user_pred_fetal
        user_df['Prediction Probability'] = np.round(user_prob_fetal, 1).astype(str) + '%'
        styled_df = user_df.style.applymap(color_prediction, subset=['Predicted Fetal Health'])
        st.dataframe(styled_df)

        st.header("**Model Performance and Insights**")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report","Feature Importance"])

        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_ada.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_ada.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition.")
        
        # Tab 3: Feature Importance Visualization
        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_ada.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

    elif model_choice == 'Soft Voting':
        sv_pickle = open('soft_voting_fetal.pickle', 'rb')
        clf4 = pickle.load(sv_pickle)
        sv_pickle.close()
        user_df = pd.read_csv(fetal_file)
        
        user_pred = clf4.predict(user_df)
        prob = clf4.predict_proba(user_df)

        mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
        user_pred_fetal = [mapping.get(pred, 'Unknown') for pred in user_pred]

        user_prob_fetal = prob.max(axis=1) * 100
        user_df['Predicted Fetal Health'] = user_pred_fetal
        user_df['Prediction Probability'] = np.round(user_prob_fetal, 1).astype(str) + '%'
        styled_df = user_df.style.applymap(color_prediction, subset=['Predicted Fetal Health'])
        st.dataframe(styled_df)

        st.header("**Model Performance and Insights**")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report","Feature Importance"])

        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_sv.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_sv.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))  
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition.")
        
        # Tab 3: Feature Importance Visualization
        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_sv.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

