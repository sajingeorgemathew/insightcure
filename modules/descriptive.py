import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Reusable layout styling
def apply_dark_style(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font_color="white",
        xaxis=dict(color="white", title_font=dict(color="white")),
        yaxis=dict(color="white", title_font=dict(color="white")),
    )
    return fig


# ============================================
# HISTOGRAM
# ============================================
def plot_histogram(df, column):
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="box",
        opacity=0.85,
        title=f"Histogram: {column}",
        template="plotly_dark",
    )
    apply_dark_style(fig)
    st.plotly_chart(fig, width="stretch")


# ============================================
# BOXPLOT
# ============================================
def plot_boxplot(df, column):
    fig = px.box(
        df,
        y=column,
        points="all",
        title=f"Boxplot: {column}",
        template="plotly_dark",
    )
    apply_dark_style(fig)
    st.plotly_chart(fig, width="stretch")


# ============================================
# CORRELATION HEATMAP (NO TRANSPARENCY)
# ============================================
def plot_corr_heatmap(df, cols):
    corr = df[cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            reversescale=True,
        )
    )

    # Heatmap should NOT be transparent
    fig.update_layout(
        title="Correlation Heatmap",
        template="plotly_dark",
        height=500,
        plot_bgcolor="rgba(30,30,30,1)",
        paper_bgcolor="rgba(30,30,30,1)",
        font_color="white",
        title_font_color="white",
    )

    st.plotly_chart(fig, width="stretch")


# ============================================
# CATEGORY VS NUMERIC HEATMAP
# ============================================
def plot_category_numeric_heatmap(df, category_col, numeric_col, agg_func):
    grouped = df.groupby(category_col)[numeric_col].agg(agg_func).reset_index()
    grouped = grouped.sort_values(numeric_col, ascending=False)

    fig = go.Figure(
        data=go.Heatmap(
            z=[grouped[numeric_col]],
            x=grouped[category_col],
            y=[f"{agg_func} of {numeric_col}"],
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title=f"{agg_func.title()} {numeric_col} by {category_col}",
        template="plotly_dark",
        height=300,
        plot_bgcolor="rgba(30,30,30,1)",
        paper_bgcolor="rgba(30,30,30,1)",
        font_color="white",
        title_font_color="white",
    )

    st.plotly_chart(fig, width="stretch")


# ============================================
# CATEGORY VS NUMERIC HEATMAP
# ============================================
def plot_category_numeric_heatmap(df, category_col, numeric_col, agg_func):
    grouped = df.groupby(category_col)[numeric_col].agg(agg_func).reset_index()
    grouped = grouped.sort_values(numeric_col, ascending=False)

    fig = go.Figure(
        data=go.Heatmap(
            z=[grouped[numeric_col]],
            x=grouped[category_col],
            y=[f"{agg_func} of {numeric_col}"],
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title=f"{agg_func.title()} {numeric_col} by {category_col}",
        template="plotly_dark",
        height=300,
        plot_bgcolor="rgba(30,30,30,1)",
        paper_bgcolor="rgba(30,30,30,1)",
        font_color="white",
        title_font_color="white",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================
# BAR CHART
# ============================================
def plot_bar(df, column):
    vc = df[column].value_counts().reset_index()
    vc.columns = [column, "Count"]

    fig = px.bar(
        vc,
        x=column,
        y="Count",
        text="Count",
        title=f"Value Counts: {column}",
        template="plotly_dark",
    )

    fig.update_traces(textposition="outside")
    apply_dark_style(fig)

    st.plotly_chart(fig, width="stretch")


# ============================================
# PIE CHART
# ============================================
def plot_pie(df, column, top_n=8):
    counts = df[column].value_counts(dropna=False)
    if len(counts) > top_n:
        top_counts = counts.head(top_n)
        other_count = counts.iloc[top_n:].sum()
        counts = pd.concat(
            [top_counts, pd.Series({"Other": other_count})]
        )

    fig = px.pie(
        values=counts.values,
        names=counts.index.astype(str),
        title=f"Distribution of {column}",
        template="plotly_dark",
    )
    apply_dark_style(fig)
    st.plotly_chart(fig, width="stretch")


# ============================================
# COLUMN CHART (AGGREGATED)
# ============================================
def plot_column_chart(df, category_col, numeric_col, agg_func):
    grouped = df.groupby(category_col)[numeric_col].agg(agg_func).reset_index()
    grouped = grouped.sort_values(numeric_col, ascending=False)

    fig = px.bar(
        grouped,
        x=category_col,
        y=numeric_col,
        title=f"{agg_func.title()} {numeric_col} by {category_col}",
        template="plotly_dark",
    )
    apply_dark_style(fig)
    st.plotly_chart(fig, width="stretch")
