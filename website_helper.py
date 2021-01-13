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
                    'padding': '5px 5px 5px 20px', 
                    'textOverflow': 'clip',
                    'overflow': 'hidden',
                    'maxWidth': '75px', 
                    'minWidth': '75px',
                    'width': '75px',
                    'minHeight': '45px', 
                    'height': '45px', 
                },
                style_cell_conditional=[
                    {
                        'if': {
                            'column_id': 'Title',
                        },
                            'minWidth': '250px',
                            'maxWidth': '250px',
                            'width': '250px',
                            'fontWeight': 'bold'
                    },
                    {
                        'if': {
                            'column_id': 'Requirements',
                        },
                            'minWidth': '250px',
                            'maxWidth': '250px',
                            'width': '250px',
                    },
                    {
                        'if': {
                            'column_id': 'Description',
                        },
                            'minWidth': '250px',
                            'maxWidth': '250px',
                            'width': '250px',
                    },
                ],
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
                },
                tooltip_data=[
                    {
                        column: {'value': str(value)}
                        for column, value in row.items()
                    } for row in recommendations.to_dict('rows')
                ],
                tooltip_duration=None,
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
                        'backgroundColor': '#c9eaf7 ',
                    },
                    {
                        'if': {
                            'column_id': 'Title',
                        },
                        'minWidth': '200px',
                        'maxWidth': '200px',
                        'width': '200px',
                        'fontWeight': 'bold',
                    },
                    {
                        'if': {
                            'column_id': 'Requirements',
                        },
                            'minWidth': '200px',
                            'maxWidth': '200px',
                            'width': '200px',
                    },
                    {
                        'if': {
                            'column_id': 'Description',
                        },
                            'minWidth': '200px',
                            'maxWidth': '200px',
                            'width': '200px',
                    },
                ],
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'Open Sans',
                    'fontSize': 13,
                    'backgroundColor': '#d9ecf3',
                    'padding': '5px 5px 5px 20px', 
                    'textOverflow': 'clip',
                    'overflow': 'hidden',
                    'maxWidth': '75px', 
                    'minWidth': '75px',
                    'width': '75px',
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
                },
                tooltip_data=[
                    {
                        column: {'value': str(value)}
                        for column, value in row.items()
                    } for row in recommendations.to_dict('rows')
                ],
                tooltip_duration=None,
            ))

