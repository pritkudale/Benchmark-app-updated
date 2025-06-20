import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the CSV file back into a DataFrame
df = pd.read_csv("ai_model_benchmarks.csv")

# Fill missing cached input costs with 0
df["Cached Input Cost"] = df["Cached Input Cost"].fillna(0)

# Calculate total cost (Input + Output)
df["Total Cost"] = df["Input Cost"] + df["Output Cost"]

# Replace NaN with None for JavaScript compatibility
df = df.where(pd.notna(df), None)

# Define benchmark options and cost options
benchmark_options = [
    {"label": "MMLU-Pro (Reasoning & Knowledge)", "value": "MMLU-Pro"},
    {"label": "GPQA Diamond (Scientific Reasoning)", "value": "GPQA Diamond"},
    {"label": "Humanity's Last Exam (Reasoning & Knowledge)", "value": "Humanity's Last Exam"},
    {"label": "LiveCodeBench (Coding)", "value": "LiveCodeBench"},
    {"label": "SciCode (Coding)", "value": "SciCode"},
    {"label": "HumanEval (Coding)", "value": "HumanEval"},
    {"label": "MATH-500 (Quantitative Reasoning)", "value": "MATH-500"},
    {"label": "AIME 2024 (Competition Math)", "value": "AIME 2024"},
    {"label": "Multilingual Index (Artificial Analysis)", "value": "Multilingual Index"},
    {"label": "LegalBench Benchmark (Legal Analysis)", "value": "LegalBench"}
]

cost_options = [
    {"label": "Input Cost", "value": "Input Cost"},
    {"label": "Output Cost", "value": "Output Cost"},
    {"label": "Total Cost", "value": "Total Cost"}
]

# Create Dash app
app = dash.Dash(
        __name__,
        meta_tags=[{"name":"viewport", "content": "width=device-width, initial-scale=1"}]
        )

server = app.server

app.title = "AI Model Benchmark Analysis"

app.layout = html.Div([
    html.Div([
    html.Img(src="assets/logo.png",width="100",height="100"),
    html.H1("AI Model Benchmark vs. Cost Analysis", style={"textAlign": "center", "marginBottom": "30px"}),
    html.Img(src="assets/logo.png",width="100",height="100"),
    ], style={"width":"100%","display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
    
    html.Div([
        html.Div([
            html.Label("Select Benchmark:"),
            dcc.Dropdown(
                id='benchmark-dropdown',
                options=benchmark_options,
                value='MMLU-Pro',
                clearable=False
            ),
        ], style={"width": "48%", "display": "inline-block"}),
        
        html.Div([
            html.Label("Select Cost Metric:"),
            dcc.Dropdown(
                id='cost-dropdown',
                options=cost_options,
                value='Total Cost',
                clearable=False
            ),
        ], style={"width": "48%", "display": "inline-block", "float": "right"})
    ], style={"marginBottom": "30px"}),
    
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('benchmark-dropdown', 'value'),
     Input('cost-dropdown', 'value')]
)
def update_graph(selected_benchmark, selected_cost):
    # Filter out rows with missing values for the selected columns
    filtered_df = df.dropna(subset=[selected_benchmark, selected_cost])
    
    # Get provider colors
    providers = filtered_df['Provider'].unique()
    provider_colors = {
        'Azure': '#0078D4',
        'AWS': '#FF9900',
        'GCP': '#008000',
        None: '#808080'
    }
    
    # Create the scatter plot with exponential x-axis
    fig = px.scatter(
        filtered_df,
        x=selected_cost,
        y=selected_benchmark,
        color='Provider',
        hover_name='Model',
        log_x=True,  # Use logarithmic scale for x-axis
        size=[10] * len(filtered_df),  # Consistent point size
        color_discrete_map=provider_colors,
        text="Model",
        height=600
    )
    
    # Update layout for better appearance
    fig.update_layout(
        title=f"{selected_benchmark} vs. {selected_cost}",
        xaxis_title=f"{selected_cost} (USD)",
        yaxis_title=f"{selected_benchmark} Score (%)",
        legend_title="Cloud Provider",
        font=dict(size=14),
        hovermode="closest",
        margin=dict(l=80, r=80, t=100, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.8)',
    )
    
    # Add model name labels to points
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=10),
        marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey'))
    )
    
    # Add grid lines
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='Grey',
        showline=True,
        linewidth=1,
        linecolor='Black'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='Grey',
        showline=True,
        linewidth=1,
        linecolor='Black'
    )
    
    # Add annotation for the best value-for-money model (highest score/cost ratio)
    if selected_cost != "Cached Input Cost":  # Skip for cached input cost
        filtered_df['value_ratio'] = filtered_df[selected_benchmark] / filtered_df[selected_cost]
        best_value_idx = filtered_df['value_ratio'].idxmax()
        best_value_model = filtered_df.loc[best_value_idx, 'Model']
        best_value_ratio = filtered_df.loc[best_value_idx, 'value_ratio']
    
    return fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8060, debug=False)
