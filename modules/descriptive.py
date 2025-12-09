import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================
#  HISTOGRAM (Interactive)
# ============================================
def plot_histogram(df, column):
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="box",
        opacity=0.85,
        title=f"Histogram: {column}",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================
#  BOXPLOT (Interactive)
# ============================================
def plot_boxplot(df, column):
    fig = px.box(
        df,
        y=column,
        points="all",
        title=f"Boxplot: {column}",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================
#  SCATTER PLOT (Interactive)
# ============================================
def plot_scatter(df, x, y):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        trendline="ols",
        title=f"Scatter Plot: {x} vs {y}",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================
#  CORRELATION HEATMAP (Interactive)
# ============================================
def plot_corr_heatmap(df, cols):
    corr = df[cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            reversescale=True
        )
    )

    fig.update_layout(
        title="Correlation Heatmap",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================
#  BAR CHART (Value Counts)
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
        template="plotly_dark"
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)


# ============================================
#  PAIRPLOT — Upgraded to Plotly Scatter Matrix
# ============================================
def plot_pairplot(df, cols):
    if len(cols) < 2:
        st.warning("Select at least two columns for pairplot.")
        return

    fig = px.scatter_matrix(
        df[cols],
        dimensions=cols,
        title="Scatter Matrix (Pairplot Alternative)",
        template="plotly_dark"
    )

    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=800)

    st.plotly_chart(fig, use_container_width=True)
