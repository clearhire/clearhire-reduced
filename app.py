import os

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
import json
import base64
from data_manipulation import load_data, random_jobs, map_jobs, map_users
from user_cf import user_cf_recommend_jobs, user_information
from item_cf import job_cf_recommend_jobs, item_cf_map_jobs
from mf_model import mf_recommend_jobs, mf_map_jobs
from website_helper import generate_table, generate_table_without_explanations
from database_explanation import db_explanation_map_jobs

job_hashmap, user_hashmap, job_ids, user_ids = load_data()
sample_jobs = random_jobs()

logo = 'logo.png'
encoded_image = base64.b64encode(open(logo, 'rb').read())

external_stylesheets = ['https:codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(style={'backgroundColor': '#3c73a8'}, children=[
    html.Div(
        id='header', 
        children=[
            html.Div(
                children=[
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    height=200)
                ], 
                style={
                    'margin': '0px 0px 0px 465px'
                }
            ),
            html.Div(
                style={
                    'textAlign': 'center',
                    'color': '#FFFFFF',
                    'fontSize': 70,
                }, 
                children='''ClearHire'''
            ),
            html.Div(
                style={
                    'textAlign': 'center',
                    'color': '#FFFFFF',
                    'fontSize': 20,
                    'font-family': 'Open Sans',
                },
                children='A platform to experiment with different styles of explanations and algorithms for job sourcing sites',
            ),
        ], 
        style={'padding': 50}
    ),

    html.Div(
        style={
            'margin': '50px 22px 0px 25px',
            'color': '#FFFFFF',
            'fontSize': 16,
            'font-family': 'Open Sans',
        }, 
        children='''
            Select (at least 4) jobs from the list that you would be interested in applying to, click SUBMIT, and then scroll down to see your results.
            The recommendation algorithms will use your selections to recommend jobs.'''
    ),

    html.Div(
        style={
            'margin': '0px 22px 0px 25px',
            'color': '#FFFFFF',
            'fontSize': 16,
            'font-family': 'Open Sans',
        }, 
        children='''
            Hover over the text to see more information'''
    ),
    
    dash_table.DataTable(
        id='sample-jobs',
        columns=[
            {"name": 'Title', "id": 'Title'}, 
            {"name": 'Description', "id": 'Description'}, 
            {"name": 'Requirements', "id": 'Requirements'}, 
            {"name": 'City', "id": 'City'}, 
            {"name": 'State', "id": 'State'}, 
            {"name": 'Country', "id": 'Country'}
        ], 
        data=sample_jobs.to_dict('records'),
        row_selectable='multi',
        selected_rows=[],
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
            'maxHeight': '45px', 
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
    ),

    html.Div(children=[
        html.Button(
            id='submit-button', 
            n_clicks=0, 
            children='Submit',
            style={
                'backgroundColor': '#FFFFFF',
                'font-family': 'Open Sans',
            }
        )],
        style={
            'padding': '22px',
        }
    ),
    

    html.Div(
        id = 'output-div',
        children=[
            html.Div(
                style={
                    'textAlign': 'left',
                    'color': '#FFFFFF',
                    'fontSize': 40,
                    'font-family': 'Open Sans',
                    'margin': '50px 0px 0px 25px'
                }, 
                children='''Your Results:'''
            ),

            html.Div(
                style={
                    'textAlign': 'left',
                    'color': '#FFFFFF',
                    'margin': '50px 22px 0px 25px',
                    'font-family': 'Open Sans',
                    'fontSize': 16,
                }, 
                children='''Below are four different algorithm and explanation combinations. Options 1 & 2 both use the same algorithm but give 
                different explanations for the results. Hence there are a total of three different recommendation algorithms being used. Option 3 and Option 4 both start by 
                explaining how the algorithm produces its results. Note that the table given in Option 3 has no 'Explanations' column. '''
            ),

            html.Div(children=[
                dcc.Dropdown(
                    id = 'tables-options',
                    options=[
                        {'label': 'Option 1', 'value': 'mf-job-explanation'},
                        {'label': 'Option 2', 'value': 'mf-db-explanation'},
                        {'label': 'Option 3', 'value': 'ucf'},
                        {'label': 'Option 4', 'value': 'icf'}
                    ],
                    value='mf-job-explanation',
                )],
                style={
                    'padding': '22px',
                    'width': '30%'
                }
            ),
            dcc.Loading(
                id='loading-state',
                type='default',
                children=html.Div(id='output-state')
            ),
        ], 
        style={
            'display': 'none',
        }
    ),
])


@app.callback(
    Output('sample-jobs', 'style_data_conditional'),
    [Input('sample-jobs', 'selected_rows')]
)
def update_styles(selected_rows):
    return [{
        'if': { 'row_index': i },
        'background_color': '#D2F3FF'
    } for i in selected_rows]


@app.callback(
    Output('output-div', 'style'),
    [Input('submit-button', 'n_clicks')]
)
def show_checklist(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-button' in changed_id:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('output-state', 'children'),
    [Input('tables-options', 'value')],
    [Input('submit-button', 'n_clicks')],
    [State('sample-jobs', 'selected_rows')], 
)
def display_tables(value, n_clicks, selected_rows):
    if (selected_rows == []):
        return None
    else: 
        selected_jobs = sample_jobs.iloc[selected_rows]
        mf_recommendations, explanations = mf_recommend_jobs(selected_jobs, job_hashmap)
        if (value == 'mf-job-explanation'):
            return(generate_table(mf_map_jobs(mf_recommendations, explanations)))
        elif (value == 'mf-db-explanation'):
            return(generate_table(db_explanation_map_jobs(mf_recommendations)))
        elif (value == 'icf'):
            job_cf_recommendations, explanation = job_cf_recommend_jobs(job_hashmap, selected_jobs)
            return( html.Div(
                        id='icf_intro_description', 
                        style={
                            'margin': '22px 30px 0px 30px',
                            'color': '#ffffff',
                            'font-family': 'Open Sans',
                            'fontSize': 16,
                            'border': '2px white solid',
                            'padding': '5px 0px 5px 10px',
                        },
                        children=['Two jobs are considered similar if many users who applied to one also applied to the other. The system recommends you similar jobs to those you selected.']
                    ),
                    generate_table(item_cf_map_jobs(job_cf_recommendations, explanation)) )
        else:
            user_cf_recommendations, nearest_neighbours = user_cf_recommend_jobs(user_hashmap, job_hashmap, selected_jobs)
            intro_description = user_information(nearest_neighbours)
            return( html.Div(
                        id='ucf_intro_description', 
                        style={
                            'margin': '22px 30px 0px 30px',
                            'color': '#ffffff',
                            'font-family': 'Open Sans',
                            'fontSize': 16,
                            'border': '2px white solid',
                            'padding': '5px 0px 5px 10px',
                        },
                        children=intro_description
                    ),   
                    generate_table_without_explanations(map_jobs(user_cf_recommendations)) )


if __name__ == '__main__':
    app.run_server(debug=True)
