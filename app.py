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

# Replace NaN with None for JavaScript compatibility (and Plotly handling)
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
# Dash will automatically serve files from an 'assets' folder
app = dash.Dash(
        __name__,
        meta_tags=[{"name":"viewport", "content": "width=device-width, initial-scale=1"}]
        )

server = app.server
app.title = "AI Model Benchmark Analysis"

app.layout = html.Div(className='app-container', children=[
    html.Div(className='header', children=[
        html.Img(src=app.get_asset_url("logo.png")), # Use app.get_asset_url for robustness
        html.H1("AI Model Benchmark vs. Cost Analysis"),
        html.Img(src=app.get_asset_url("logo.png"))
    ]),
    
    html.Div(className='controls-container', children=[
        html.Div(className='control-item', children=[
            html.Label("Select Benchmark:"),
            dcc.Dropdown(
                id='benchmark-dropdown',
                options=benchmark_options,
                value='MMLU-Pro',
                clearable=False
            ),
        ]),
        
        html.Div(className='control-item', children=[
            html.Label("Select Cost Metric:"),
            dcc.Dropdown(
                id='cost-dropdown',
                options=cost_options,
                value='Total Cost',
                clearable=False
            ),
        ])
    ]),
    
    html.Div(className='graph-container', children=[
        dcc.Graph(id='scatter-plot', config={'displayModeBar': True, 'scrollZoom': True})
    ])
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('benchmark-dropdown', 'value'),
     Input('cost-dropdown', 'value')]
)
def update_graph(selected_benchmark, selected_cost):
    # Filter out rows with missing values for the selected columns
    # Also ensure benchmark scores are numeric
    temp_df = df.copy()
    temp_df[selected_benchmark] = pd.to_numeric(temp_df[selected_benchmark], errors='coerce')
    temp_df[selected_cost] = pd.to_numeric(temp_df[selected_cost], errors='coerce')
    
    filtered_df = temp_df.dropna(subset=[selected_benchmark, selected_cost])
    
    # Filter out non-positive costs if log scale is used for cost
    if selected_cost != "Cached Input Cost": # Assuming Cached Input Cost might not always be log
        filtered_df = filtered_df[filtered_df[selected_cost] > 0]

    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {selected_benchmark} and {selected_cost}",
            font=dict(family="Poppins", size=14, color="#4b5563"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#ffffff',
            height=650, # Match graph height
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                dict(
                    text="No data to display for the selected criteria.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(family="Poppins",size=18,color="#6b7280")
                )
            ]
        )
        return fig

    providers = filtered_df['Provider'].unique()
    provider_colors = {
        'Azure': '#0078D4',  # Microsoft Blue
        'AWS': '#FF9900',    # Amazon Orange
        'GCP': '#34A853',    # Google Green (one of them)
        None: '#808080',     # Grey for unspecified
        'OpenAI': '#10A37F', # OpenAI Teal
        'Anthropic': '#D95C3F', # Anthropic Salmon/Red
        'Mistral AI': '#FF4F00', # Mistral Orange
        'Google': '#4285F4', # Google Blue
        'Meta': '#0068FF', # Meta Blue
        # Add more providers and their official/representative colors if needed
    }
    # Default color for any provider not in the map
    default_color = '#BDBDBD' # A neutral grey
    for p in providers:
        if p not in provider_colors:
            provider_colors[p] = default_color
            
    fig = px.scatter(
        filtered_df,
        x=selected_cost,
        y=selected_benchmark,
        color='Provider',
        hover_name='Model',
        hover_data={
            selected_cost: ':.4f', # Format cost (e.g., $0.0015)
            selected_benchmark: ':.2f',
            'Provider': True,
            'Model': True,
            'Input Cost': ':.4f',
            'Output Cost': ':.4f',
            'Total Cost': ':.4f'
        },
        log_x=True,
        color_discrete_map=provider_colors,
        text="Model"
    )
    
    fig.update_layout(
        title=dict(
            text=f"{selected_benchmark} vs. {selected_cost}",
            font=dict(family="Poppins", size=24, color="#1f2937"), # Darker, larger title
            x=0.5,
            xanchor='center',
            pad=dict(t=10, b=20) # Padding for title
        ),
        xaxis_title=f"{selected_cost} (USD, log scale)",
        yaxis_title=f"{selected_benchmark} Score (%)",
        font=dict(family="Poppins", size=12, color="#4b5563"), # Default font for non-specified elements
        xaxis=dict(
            title_font=dict(family="Poppins", size=16, color="#374151"),
            tickfont=dict(family="Poppins", size=11, color="#6b7280"),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(229, 231, 235, 0.9)', # Slightly more visible grid
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(209, 213, 219, 1)' # Axis line color
        ),
        yaxis=dict(
            title_font=dict(family="Poppins", size=16, color="#374151"),
            tickfont=dict(family="Poppins", size=11, color="#6b7280"),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(229, 231, 235, 0.9)',
            zeroline=True, # Keep zeroline for score axis if it makes sense
            zerolinewidth=1,
            zerolinecolor='rgba(209, 213, 219, 1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(209, 213, 219, 1)'
        ),
        legend=dict(
            title_text="Cloud Provider / Model Source",
            title_font=dict(family="Poppins", size=14, color="#374151"),
            font=dict(family="Poppins", size=12, color="#4b5563"),
            bgcolor="rgba(255,255,255,0.85)", # Semi-transparent white
            bordercolor="rgba(229, 231, 235, 0.9)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02, # Position above plot
            xanchor="center", # Center legend
            x=0.5
        ),
        hovermode="closest",
        margin=dict(l=80, r=40, t=120, b=80), # Increased top margin for legend and title
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#ffffff', # Clean white plot area
        height=650 # Taller graph
    )
    
    fig.update_traces(
        textposition='top center',
        textfont=dict(family="Poppins", size=9, color="#555555"), # Slightly darker text
        marker=dict(size=11, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')),
        # selector=dict(type='scatter') # Ensure it applies to scatter traces
    )
    
    # --- BEST VALUE ANNOTATION REMOVED ---
    # The block of code for calculating and adding the best value annotation
    # and highlighting has been removed from here.
    
    return fig

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8060, debug=False)