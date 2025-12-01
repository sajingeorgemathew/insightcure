import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Histogram: {column}")
    st.pyplot(fig)

def plot_boxplot(df, column):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot: {column}")
    st.pyplot(fig)

def plot_scatter(df, x, y):
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x], y=df[y], ax=ax)
    ax.set_title(f"Scatter Plot: {x} vs {y}")
    st.pyplot(fig)

def plot_corr_heatmap(df, cols):
    fig, ax = plt.subplots(figsize=(8,5))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

def plot_bar(df, column):
    fig, ax = plt.subplots()
    df[column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Value Counts: {column}")
    st.pyplot(fig)

def plot_pairplot(df, cols):
    st.write("Generating Pairplot… may take a moment")
    fig = sns.pairplot(df[cols])
    st.pyplot(fig)
