from dash import Dash, dash_table, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from utils import *

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[{'name': 'viewport',
                        'content': 'width=device-width, initial-scale=1.0'}]
            )

df = pd.read_csv('./res.csv')
# prev = df.copy()

columns = list(df.columns)

app.layout = html.Div([
    html.H1('Confidence Dashboard',className="text-center text-primary m-4",style={'font-family':'Open Sans'}),
    dash_table.DataTable(
        id='table-editing',
        columns=[{'id': c, 'name': c, 'editable':check_editable(c)} for c in columns],
        data=df.to_dict(orient='records'),
    ),
])

@app.callback(
    Output(component_id='table-editing', component_property='data'),
    Input('table-editing', 'data'),
    Input('table-editing', 'columns'))
def display_output(rows, columns):
    curr_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    # global prev
    # col_changed = find_change_column(curr_df,prev)
    # if col_changed is None:
    #     print('No column changed')
    #     return curr_df.to_dict(orient='records')
    
    prev = curr_df.copy()
    # loose the confidence thing
    curr_df = curr_df.drop(columns=['final_confidence'],axis=1)
    
    new_df = perform_operation(curr_df)
    
    return new_df.to_dict(orient='records')



if __name__ == '__main__':
    app.run_server(debug=True)
