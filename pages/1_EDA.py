import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Titanic EDA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Titanic Dataset Analysis")

st.markdown("---")

df = pd.read_csv("titanic.csv")

st.subheader("Dataset Preview")

st.dataframe(df)

st.markdown("---")

col1,col2,col3=st.columns(3)

with col1:

    st.metric(
        "Passengers",
        len(df)
    )

with col2:

    st.metric(
        "Survived",
        df["Survived"].sum()
    )

with col3:

    st.metric(
        "Not Survived",
        len(df)-df["Survived"].sum()
    )

st.markdown("---")

tab1,tab2,tab3,tab4,tab5=st.tabs(
    [
        "Survival",
        "Gender",
        "Passenger Class",
        "Fare",
        "Correlation"
    ]
)

with tab1:

    fig=px.pie(
        df,
        names="Survived",
        title="Survival Distribution",
        color="Survived",
        color_discrete_sequence=["red","green"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    fig=px.histogram(
        df,
        x="Sex",
        color="Survived",
        barmode="group",
        title="Gender Wise Survival"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab3:

    fig=px.histogram(
        df,
        x="Pclass",
        color="Survived",
        barmode="group",
        title="Passenger Class Survival"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab4:

    fig=px.box(
        df,
        y="Fare",
        color="Survived",
        title="Fare Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab5:

    data=df.copy()

    data["Sex"]=data["Sex"].map(
        {
            "male":0,
            "female":1
        }
    )

    data["Embarked"]=data["Embarked"].map(
        {
            "C":0,
            "Q":1,
            "S":2
        }
    )

    corr=data.corr(numeric_only=True)

    fig=ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        showscale=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

st.subheader("Dataset Information")

buffer = io.StringIO()

df.info(buf=buffer)

info = buffer.getvalue()

st.text(info)