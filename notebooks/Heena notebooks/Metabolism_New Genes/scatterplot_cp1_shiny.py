# among conditions, choose a final sim flux that is closet to target flux, report its condition as well.
# if a reaction chosen has same flux as basal, then condition defaults to basal
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load the data
df_kc_cp1 = pd.read_csv("df_kc_cp1.csv")
conditions_names = ['sim_cp1_basal','sim_cp1_Isoprimeverose','sim_cp1_Oxamate','sim_cp1_Trehalose','sim_cp1_GLC-1-P',
                    'sim_cp1_3-HYDROXYPHENYL-PROPIONATE','sim_cp1_3-PHENYLPROPIONATE','sim_cp1_cinnate','sim_cp1_L-galactonate',
                    'sim_cp1_D-GLUCARATE']

app_ui = ui.page_fluid(
    ui.h2("Simulated vs Target Kinetic Reactions Fluxome"),
    ui.input_select("xcol", "Choose Simulation Condition", {c: c for c in conditions_names}),
    ui.input_checkbox("color_by_new", "New vs Old Reactions:", value=True),
    output_widget("scatter_plot", width="100%", height="600px"),
)

def server(input, output, session):

    # @output
    @render_widget
    def scatter_plot():
        data = df_kc_cp1
        condition_selected = input.xcol()
        color_col = input.color_by_new()
        if color_col:
            fig = px.scatter(
                data,
                x="kinetic",
                y=condition_selected,
                color='is_new',
                color_discrete_map={
                    "Old Reactions": "purple",
                    "New Reactions": "orange"
                },
                opacity=0.7,
                hover_name=data.index
            )
        else:
            fig = px.scatter(
                data,
                x=condition_selected,
                y="kinetic",
                opacity=0.7,
                hover_name=data.index)

        fig.add_trace(go.Scatter(
            x=data["kinetic"],
            y=data["kinetic"],
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
        return fig


app = App(app_ui, server)