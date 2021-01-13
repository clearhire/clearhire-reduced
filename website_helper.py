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
                    'whiteSpace': 'nowrap',
                },
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'Open Sans',
                    'fontSize': 13,
                    'backgroundColor': '#d9ecf3',
                    'padding': '5px', 
                    'textOverflow': 'ellipsis',
                    'overflow': 'hidden',
                    'maxWidth': '175px', 
                    'minWidth': '175px',
                    'width': '175px',
                    'minHeight': '45px', 
                    'height': '45px', 
                },
                style_cell_conditional=[{
                    'if': {
                        'column_id': 'Title',
                    },
                        'minWidth': '270px',
                        'maxWidth': '270px',
                        'width': '270px',
                        'fontWeight': 'bold'
                }],
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#a7dff3e8',
                    'fontSize': 16,
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
                style_cell_conditional=[
                    {
                        'if': {
                            'column_id': 'Explanations',
                        },
                        'minWidth': '235px',
                        'maxWidth': '235px',
                        'width': '235px',
                        'whiteSpace': 'normal',
                        'backgroundColor': '#c9eaf7',
                    },
                    {
                        'if': {
                            'column_id': 'Title',
                        },
                        'minWidth': '205px',
                        'maxWidth': '205px',
                        'width': '205px',
                        'fontWeight': 'bold',
                    },
                ],
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'Open Sans',
                    'fontSize': 13,
                    'backgroundColor': '#d9ecf3',
                    'padding': '5px', 
                    'textOverflow': 'ellipsis',
                    'overflow': 'hidden',
                    'maxWidth': '150px', 
                    'minWidth': '150px',
                    'width': '150px',
                    'minHeight': '45px', 
                    'maxHeight': '45px', 
                    'height': '45px', 
                },
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#a7dff3e8',
                    'fontSize': 16,
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

