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
                children='A platform to experiment with different styles of explanations and algorithms for job listing sites',
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
        children=['You want to apply for a new job. ClearHire allows you to explore different algorithms and explanations for job recommendations to find the one that suits you best. To get started:', 
                    html.Br(), html.Br(), 'Step 1: Select at least four jobs from the list below that you would be interested in applying to. Hover your mouse over the text for more information about each one.',
                    html.Br(), 'Step 2: Click SUBMIT',
                    html.Br(), 'Step 3: Scroll down to see four different lists of job recommendationss, based on the jobs you selected']
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
            'padding': '5px 5px 0px 20px', 
            'textOverflow': 'clip',
            'overflow': 'hidden',
            'maxWidth': '75px', 
            'minWidth': '75px',
            'width': '75px',
            'minHeight': '45px', 
            'maxHeight': '45px', 
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
            } for row in sample_jobs.to_dict('rows')
        ],
        tooltip_duration=None,
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
                children=['Below are the four list of recommendations based on your selections, each with a different algorithm/explanation combination.',
                            html.Br(), 'Options 1 & 2 both use the same algorithm but give different explanations. Hence there are a total of three different algorithms being used.',
                            html.Br(), 'Options 3 & 4 both start by explaining how the algorithm produces its results. Note that the table given in Option 3 has no Explanations column.']
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
