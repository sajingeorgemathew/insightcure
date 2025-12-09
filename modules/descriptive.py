import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Helper: Apply uniform dark styling (transparent background)
# ============================================================
def _apply_dark_layout(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",   # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",    # Transparent plot area
        font=dict(color="white"),
        title_font=dict(color="white", size=20),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    return fig


# ============================================================
# HISTOGRAM (Interactive)
# ============================================================
def plot_histogram(df, column):
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        opacity=0.85,
        marginal="box",
    )
    _apply_dark_layout(fig, f"Histogram: {column}")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BOXPLOT (Interactive)
# ============================================================
def plot_boxplot(df, column):
    fig = px.box(
        df,
        y=column,
        points="all",
    )
    _apply_dark_layout(fig, f"Boxplot: {column}")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SCATTER PLOT (Interactive)
# ============================================================
def plot_scatter(df, x, y):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        trendline="ols",
    )
    _apply_dark_layout(fig, f"Scatter Plot: {x} vs {y}")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CORRELATION HEATMAP (Interactive)
# ============================================================
def plot_corr_heatmap(df, cols):
    corr = df[cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            reversescale=True,
            zmin=-1, zmax=1,
            hoverongaps=False
        )
    )

    _apply_dark_layout(fig, "Correlation Heatmap")
    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BAR CHART (Value Counts)
# ============================================================
def plot_bar(df, column):
    vc = df[column].value_counts().reset_index()
    vc.columns = [column, "Count"]

    fig = px.bar(
        vc,
        x=column,
        y="Count",
        text="Count",
    )

    fig.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="white", width=1))
    )

    _apply_dark_layout(fig, f"Value Counts: {column}")

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAIRPLOT — Plotly Scatter Matrix
# ============================================================
def plot_pairplot(df, cols):
    if len(cols) < 2:
        st.warning("Select at least two columns for pairplot.")
        return

    fig = px.scatter_matrix(
        df,
        dimensions=cols,
    )

    fig.update_traces(diagonal_visible=False)

    _apply_dark_layout(fig, "Scatter Matrix (Pairplot Alternative)")

    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
