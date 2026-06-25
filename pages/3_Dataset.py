import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Titanic Dataset",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Titanic Dataset Explorer")

st.markdown("---")

df = pd.read_csv("titanic.csv")

# ===========================
# Sidebar Filters
# ===========================

st.sidebar.header("🔍 Filter Dataset")

gender = st.sidebar.multiselect(
    "Gender",
    options=df["Sex"].unique(),
    default=df["Sex"].unique()
)

pclass = st.sidebar.multiselect(
    "Passenger Class",
    options=sorted(df["Pclass"].unique()),
    default=sorted(df["Pclass"].unique())
)

survived = st.sidebar.multiselect(
    "Survived",
    options=[0,1],
    default=[0,1]
)

filtered_df = df[
    (df["Sex"].isin(gender)) &
    (df["Pclass"].isin(pclass)) &
    (df["Survived"].isin(survived))
]

# ===========================
# Metrics
# ===========================

col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric("Rows", len(filtered_df))

with col2:
    st.metric("Columns", len(filtered_df.columns))

with col3:
    st.metric(
        "Survived",
        filtered_df["Survived"].sum()
    )

with col4:
    st.metric(
        "Not Survived",
        len(filtered_df)-filtered_df["Survived"].sum()
    )

st.markdown("---")

# ===========================
# Search
# ===========================

search = st.text_input(
    "🔍 Search Passenger Name"
)

if search:

    filtered_df = filtered_df[
        filtered_df["Name"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

# ===========================
# Dataset
# ===========================

st.subheader("Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# ===========================
# Missing Values
# ===========================

st.markdown("---")

st.subheader("Missing Values")

missing = pd.DataFrame({
    "Column":df.columns,
    "Missing Values":df.isnull().sum().values
})

st.dataframe(
    missing,
    use_container_width=True
)

# ===========================
# Statistics
# ===========================

st.markdown("---")

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# ===========================
# Column Information
# ===========================

st.markdown("---")

st.subheader("Column Information")

info = pd.DataFrame({
    "Column":df.columns,
    "Datatype":df.dtypes.values
})

st.dataframe(
    info,
    use_container_width=True
)

# ===========================
# Download
# ===========================

st.markdown("---")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="📥 Download Dataset",
    data=csv,
    file_name="filtered_titanic.csv",
    mime="text/csv"
)