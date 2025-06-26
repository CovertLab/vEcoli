import pandas as pd
import plotly.express as px
import plotly.io as pio
from shiny import App, ui, render, reactive
import plotly.graph_objects as go
from htmltools import HTML

# === Load Data ===
df = pd.read_csv("df_kc_cp1_combined.csv", index_col=0)

# === Define Conditions ===
conditions_names = [
    'sim_cp1_basal','sim_cp1_Isoprimeverose','sim_cp1_Oxamate','sim_cp1_Trehalose',
    'sim_cp1_GLC-1-P', 'sim_cp1_3-HYDROXYPHENYL-PROPIONATE','sim_cp1_3-PHENYLPROPIONATE',
    'sim_cp1_cinnate','sim_cp1_L-galactonate', 'sim_cp1_D-GLUCARATE'
]

condition_choices = ['no selection'] + conditions_names

# === UI ===
app_ui = ui.page_fluid(
    ui.h2("Flux vs Target Interactive Plot"),
    ui.input_select("highlight_cond", "Select Condition to Highlight",
                           {cond: cond for cond in condition_choices}, selected="none"),
    ui.output_ui("flux_plot")
)

# === Server Logic ===
def server(input, output, session):

    @output
    @render.ui
    def flux_plot():
        selected_cond = input.highlight_cond()
        data = df.copy()
        data['index'] = data.index
        # Default coloring
        def assign_color(row):
            if row['condition'] == 'none':
                return 'No flow'
            elif row['condition'] == 'all':
                return 'Same flow through all conditions'
            elif selected_cond in row['condition']:
                return 'Condition selected'   # Highlight color for selected condition
            else:
                return 'Condition specific flow'

        data['color'] = data.apply(assign_color, axis=1)


        fig = px.scatter(
            data, 
            x='target',
            y='flux',
            color=data['color'],
            color_discrete_map={
                'No flow': 'grey',
                'Same flow through all conditions': '#dbb9b3',
                'Condition selected': 'coral',
                'Flow through other conditions': 'pink',
                'Condition specific flow': 'purple'
            },
            opacity=0.7,
            hover_data=['index', 'condition']
        )

        #Add a diagonal line
        fig.add_trace(go.Scatter(
            x=data["target"],
            y=data["target"],
            mode='lines',
            line=dict(color='grey', dash='solid', width=3),
            showlegend=False
                ))

        # Move the diagonal line behind the scatter points
        fig.data = (fig.data[-1],) + fig.data[:-1]

        fig.update_layout(
            xaxis_title='Target flux log(flux)',  # x-axis name
            yaxis_title='Simluated flux log(flux)',  # y-axis name
        )

        return HTML(pio.to_html(fig, full_html=False))

# === Run App ===
app = App(app_ui, server)
