import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

def generate_table_without_explanations(df):
    recommendations = df

    return( dash_table.DataTable(
                id='recommendations',
                columns=[
                    {"name": 'Title', "id": 'Title'}, 
                    {"name": 'Description', "id": 'Description'}, 
                    {"name": 'Requirements', "id": 'Requirements'}, 
                    {"name": 'City', "id": 'City'}, 
                    {"name": 'State', "id": 'State'}, 
                    {"name": 'Country', "id": 'Country'}
                ], 
                data=recommendations.to_dict('records'),
                style_data={
                    'whiteSpace': 'normal',
                },
                css=[{
                    'selector': '.dash-spreadsheet td div',
                    'rule': '''
                        line-height: 15px;
                        max-height: 90px; min-height: 90px; height: 90px;
                        display: block;
                        overflow-y: hidden;
                    '''
                }],
                tooltip_data=[
                    {
                        column: {'value': str(value)}
                        for column, value in row.items()
                    } for row in recommendations.to_dict('rows')
                ],
                tooltip_duration=None,
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'Arial',
                    'fontSize': 11,
                    'backgroundColor': '#d9ecf3',
                    'padding': '5px'
                },
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#a7dff3e8',
                    'fontSize': 14,
                    'minHeight': '60px',
                    'height': '60px'
                },
                style_as_list_view=True,
                style_table={
                    'border': '22px solid #3c73a8',
                    'borderRadius': '15px',
                    'width': '96%'
                }
            ))


def generate_table(df):
    recommendations = df

    return( dash_table.DataTable(
                id='recommendations',
                columns=[
                    {"name": 'Title', "id": 'Title'}, 
                    {"name": 'Description', "id": 'Description'}, 
                    {"name": 'Requirements', "id": 'Requirements'}, 
                    {"name": 'City', "id": 'City'}, 
                    {"name": 'State', "id": 'State'}, 
                    {"name": 'Country', "id": 'Country'},
                    {"name": 'Explanations', "id": 'Explanations'}
                ], 
                data=recommendations.to_dict('records'),
                style_data={
                    'whiteSpace': 'normal',
                },
                style_data_conditional=[{
                    'if': {
                        'column_id': 'Explanations',
                    },
                    'minWidth': '255px',
                    'maxWidth': '255px',
                    'width': '255px',
                    'fontWeight': 'bold'
                }],
                css=[{
                    'selector': '.dash-spreadsheet td div',
                    'rule': '''
                        line-height: 15px;
                        max-height: 90px; min-height: 90px; height: 90px;
                        display: block;
                        overflow-y: hidden;
                    '''
                }],
                tooltip_data=[
                    {
                        column: {'value': str(value)}
                        for column, value in row.items()
                    } for row in recommendations.to_dict('rows')
                ],
                tooltip_duration=None,
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'Arial',
                    'fontSize': 11,
                    'backgroundColor': '#d9ecf3',
                    'padding': '5px'
                },
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#a7dff3e8',
                    'fontSize': 14,
                    'minHeight': '60px',
                    'height': '60px'
                },
                style_as_list_view=True,
                style_table={
                    'border': '22px solid #3c73a8',
                    'borderRadius': '15px',
                    'width': '96%'
                }
            ))

