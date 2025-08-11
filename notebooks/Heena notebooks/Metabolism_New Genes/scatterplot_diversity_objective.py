import pandas as pd
import plotly.express as px
import plotly.io as pio
from shiny import App, ui, render, reactive
import plotly.graph_objects as go
from htmltools import HTML

# === Load Data ===
df = pd.read_csv("diversity_objective.csv", index_col=0)

# === Define Comparisons ===
comparison_names = [
    'default (efficiency only) versus diversity + efficiency',
    'default (efficiency only) versus diversity only',
    'diversity + efficiency versus diversity only',
    'kinetic'
]


# === UI ===
app_ui = ui.page_fluid(
    ui.h2("Effect of Modifying Objective Function on Fluxes"),
    ui.input_select("highlight_comp", "Select Comparisons to Highlight",
                           {cond: cond for cond in comparison_names}, selected="none"),
    ui.input_checkbox_group("highlight_is_new", "Select Reactions to Highlight",
                           {'New Reactions': ui.span('New Reactions'),
                                    "Old Reactions": ui.span('Old Reactions'),
                            },
                            selected=['New Reactions', 'Old Reactions']),
    ui.output_ui("flux_plot")
)

# === Server Logic ===
def server(input, output, session):

    @output
    @render.ui
    def flux_plot():
        # subset dataframe based on selection
        selected_comp = input.highlight_comp()
        selected_reactions = input.highlight_is_new()

        data = df.copy()
        data['index'] = data.index
        data = data[data['is_new'].isin(selected_reactions)]

        # Default coloring
        # def assign_color(row):
        #     if row['condition'] == 'none':
        #         return 'No flow'
        #     elif row['condition'] == 'all':
        #         return 'Same flow through all conditions'
        #     elif selected_comp in row['condition']:
        #         return 'Condition selected'   # Highlight color for selected condition
        #     else:
        #         return 'Condition specific flow'
        #
        # data['Color Index'] = data.apply(assign_color, axis=1)

        if selected_comp == 'default (efficiency only) versus diversity + efficiency':
            y_data = data['0']
            x_data = data['add diversity']

            y_label = 'default (efficiency only)'
            x_label = 'diversity + efficiency flux'
            color_y = None
            data_used = data
        elif selected_comp == 'default (efficiency only) versus diversity only':
            y_data = data['0']
            x_data = data['remove efficiency and add diversity']

            y_label = 'default (efficiency only)'
            x_label = 'diversity only flux'
            color_y = None
            data_used = data
        elif selected_comp == 'diversity + efficiency versus diversity only':
            y_data = data['add diversity']
            x_data = data['remove efficiency and add diversity']

            y_label = 'diversity + efficiency flux'
            x_label = 'diversity only flux'
            color_y = None
            data_used = data
        elif selected_comp == 'kinetic':
            data_kinetic = data[data['kinetic'] != 'False'].copy()
            data_kinetic['kinetic'] = data_kinetic['kinetic'].astype(float)
            data_melted = data_kinetic.melt(id_vars=['kinetic'],
                                            value_vars=['0', 'add diversity', 'remove efficiency and add diversity'])
            x_data = data_melted['kinetic']
            y_data = data_melted['value']

            y_label = 'simulated flux'
            x_label = 'target flux'
            color_y = data_melted['variable']
            data_used = data_melted

        fig = px.scatter(
            data_used,
            x=x_data,
            y=y_data,
            color=color_y,
            opacity=0.5,
            # hover_data=['index']
        )

        #Add a diagonal line
        fig.add_trace(go.Scatter(
            x=x_data,
            y=x_data,
            mode='lines',
            line=dict(color='grey', dash='solid', width=3),
            showlegend=False
            ))

        # Move the diagonal line behind the scatter points
        fig.data = (fig.data[-1],) + fig.data[:-1]

        fig.update_layout(
            xaxis_title=x_label,  # x-axis name
            yaxis_title=y_label,  # y-axis name
        )

        return HTML(pio.to_html(fig, full_html=False))

# === Run App ===
app = App(app_ui, server)
