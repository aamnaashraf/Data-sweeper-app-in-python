import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # Use plotly for visualizations
from ydata_profiling import ProfileReport  # Use ydata-profiling instead of pandas-profiling
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore

# App configuration
st.set_page_config(
    page_title="Data Sweeper Pro",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# Main app function
def main():
    st.sidebar.title("Navigation")
    app_pages = {
        "Upload Data": upload_data,
        "Data Profiling": data_profiling,
        "Data Cleaning": data_cleaning,
        "Advanced Processing": advanced_processing,
        "Export Data": export_data
    }
    page_selection = st.sidebar.radio("Go to", list(app_pages.keys()))
    app_pages[page_selection]()

# Page functions
def upload_data():
    st.title("📤 Data Upload")
    
    upload_option = st.radio("Choose data source:", 
                           ["Upload CSV/Excel", "Sample Dataset"])
    
    if upload_option == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            try:
                # Read file based on type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_name = st.selectbox("Select Excel sheet", excel_file.sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                
                st.session_state.raw_df = df
                st.session_state.processed_df = df.copy()
                st.success("Data uploaded successfully!")
                show_data_preview()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    else:
        sample_dataset = st.selectbox("Choose sample dataset",
                                    ["Titanic", "Iris", "Tips"])
        if sample_dataset == "Titanic":
            df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
        elif sample_dataset == "Iris":
            df = px.data.iris()  # Use plotly.express for sample datasets
        else:
            df = px.data.tips()  # Use plotly.express for sample datasets
        st.session_state.raw_df = df
        st.session_state.processed_df = df.copy()
        show_data_preview()

def show_data_preview():
    st.subheader("Data Preview")
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state.raw_df.head(100))
    
    with st.expander("Data Summary"):
        if st.session_state.raw_df is not None:
            st.write(f"Shape: {st.session_state.raw_df.shape}")
            st.write("Columns:")
            for col in st.session_state.raw_df.columns:
                dtype = str(st.session_state.raw_df[col].dtype)
                st.write(f"- {col} ({dtype})")
def data_profiling():
    st.title("🔍 Data Profiling")
    if st.session_state.raw_df is not None:
        pr = ProfileReport(st.session_state.raw_df, explorative=True)
        st.write(pr.to_html(), unsafe_allow_html=True)  # Display the report in Streamlit
    else:
        st.warning("Please upload data first!")

def data_cleaning():
    st.title("🧹 Data Cleaning Tools")
    if st.session_state.processed_df is None:
        st.warning("Please upload data first!")
        return

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Handle Missing Values")
        missing_cols = st.session_state.processed_df.columns[
            st.session_state.processed_df.isna().any()
        ].tolist()
        
        if missing_cols:
            selected_missing_col = st.selectbox("Select column with missing values:", missing_cols)
            handling_method = st.selectbox("Handling method:",
                                         ["Drop rows", "Fill with mean", 
                                          "Fill with median", "Fill with mode",
                                          "Interpolate", "Custom value"])
            
            if handling_method == "Custom value":
                custom_value = st.text_input("Enter custom fill value:")
            
            if st.button("Apply Missing Value Treatment"):
                handle_missing_values(selected_missing_col, handling_method, custom_value if 'custom_value' in locals() else None)
        else:
            st.info("No missing values detected")

    with col2:
        st.subheader("Remove Duplicates")
        dup_count = st.session_state.processed_df.duplicated().sum()
        st.write(f"Duplicate rows found: {dup_count}")
        if dup_count > 0 and st.button("Remove Duplicates"):
            st.session_state.processed_df = st.session_state.processed_df.drop_duplicates()
            st.success(f"Removed {dup_count} duplicates!")

    with col3:
        st.subheader("Data Type Conversion")
        convert_col = st.selectbox("Select column to convert:", 
                                 st.session_state.processed_df.columns)
        new_type = st.selectbox("Select new data type:",
                              ["str", "int", "float", "datetime", "category"])
        
        if st.button("Convert Data Type"):
            try:
                st.session_state.processed_df[convert_col] = st.session_state.processed_df[convert_col].astype(new_type)
                st.success(f"Converted {convert_col} to {new_type}")
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")

    st.subheader("Interactive Data Filtering")
    filter_condition = st.text_input("Enter pandas query (e.g., Age > 30):")
    if filter_condition:
        try:
            filtered_df = st.session_state.processed_df.query(filter_condition)
            st.dataframe(filtered_df)
            if st.button("Apply Filter Permanently"):
                st.session_state.processed_df = filtered_df
        except Exception as e:
            st.error(f"Invalid filter condition: {str(e)}")

def handle_missing_values(col, method, custom_value=None):
    df = st.session_state.processed_df
    if method == "Drop rows":
        df = df.dropna(subset=[col])
    elif method == "Fill with mean":
        df[col] = df[col].fillna(df[col].mean())
    elif method == "Fill with median":
        df[col] = df[col].fillna(df[col].median())
    elif method == "Fill with mode":
        df[col] = df[col].fillna(df[col].mode()[0])
    elif method == "Interpolate":
        df[col] = df[col].interpolate()
    elif method == "Custom value":
        df[col] = df[col].fillna(custom_value)
    
    st.session_state.processed_df = df
    st.success(f"Applied {method} to {col}")

def advanced_processing():
    st.title("⚙️ Advanced Processing")
    if st.session_state.processed_df is None:
        st.warning("Please upload data first!")
        return
    
    st.subheader("Feature Engineering")
    new_feature_name = st.text_input("New feature name:")
    feature_expression = st.text_input("Feature expression (e.g., col1 + col2):")
    
    if st.button("Create New Feature"):
        try:
            st.session_state.processed_df.eval(f"{new_feature_name} = {feature_expression}", inplace=True)
            st.success("Feature created successfully!")
        except Exception as e:
            st.error(f"Feature creation failed: {str(e)}")
    
    st.subheader("Outlier Detection")
    numeric_cols = st.session_state.processed_df.select_dtypes(include=np.number).columns.tolist()
    selected_outlier_col = st.selectbox("Select numeric column:", numeric_cols)
    
    if selected_outlier_col:
        fig = px.box(st.session_state.processed_df, y=selected_outlier_col)  # Use plotly for visualization
        st.plotly_chart(fig)
        
        outlier_method = st.selectbox("Outlier detection method:",
                                     ["Z-Score", "IQR"])
        threshold = st.slider("Threshold:", 1.0, 5.0, 3.0)
        
        if st.button("Detect & Handle Outliers"):
            handle_outliers(selected_outlier_col, outlier_method, threshold)

def handle_outliers(col, method, threshold):
    df = st.session_state.processed_df
    original_count = len(df)
    
    if method == "Z-Score":
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        df = df[abs(z_scores) < threshold]
    elif method == "IQR":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - threshold * IQR)) | 
                 (df[col] > (Q3 + threshold * IQR)))]
    
    removed_count = original_count - len(df)
    st.session_state.processed_df = df
    st.success(f"Removed {removed_count} outliers using {method} method")

def export_data():
    st.title("📤 Export Cleaned Data")
    if st.session_state.processed_df is not None:
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.processed_df.head())
        
        export_format = st.selectbox("Select export format:",
                                   ["CSV", "Excel", "Pickle"])
        
        filename = st.text_input("Enter file name:", "cleaned_data")
        
        if st.button("Export Data"):
            try:
                if export_format == "CSV":
                    csv = st.session_state.processed_df.to_csv(index=False).encode()
                    st.download_button(label="Download CSV",
                                      data=csv,
                                      file_name=f"{filename}.csv",
                                      mime="text/csv")
                elif export_format == "Excel":
                    excel_file = st.session_state.processed_df.to_excel(index=False)
                    st.download_button(label="Download Excel",
                                      data=excel_file,
                                      file_name=f"{filename}.xlsx",
                                      mime="application/vnd.ms-excel")
                elif export_format == "Pickle":
                    pickle_data = st.session_state.processed_df.to_pickle()
                    st.download_button(label="Download Pickle",
                                       data=pickle_data,
                                       file_name=f"{filename}.pkl",
                                       mime="application/octet-stream")
                st.success("Data exported successfully!")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    else:
        st.warning("No processed data to export!")

if __name__ == "__main__":
    main()