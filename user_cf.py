import pandas as pd
import numpy as np
import math
import dash_html_components as html
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors
from data_manipulation import fit_cf_model

def similar_users(user_job_mat, user_mapper, job_mapper, selected_jobs, n=4):
    '''takes matrix and map from job_id to index in matrix, and job_id and returns the n most similar jobs'''
    alpha = 40

    selected_job_ids = [row["JobID"] for _, row in selected_jobs.iterrows()]
    selected_job_indices = [job_mapper[job_id] for job_id in selected_job_ids]

    #for some reason size of user-job-matrix increases at every iteration so this reduces it back to size
    user_job_mat = user_job_mat[:9123, :]
    n_users, n_jobs = user_job_mat.shape

    ratings = [alpha for i in range(len(selected_job_indices))]

    user_job_new_mat = user_job_mat
    user_job_new_mat.data = np.hstack((user_job_mat.data, ratings))
    user_job_new_mat.indices = np.hstack((user_job_mat.indices, selected_job_indices))
    user_job_new_mat.indptr = np.hstack((user_job_mat.indptr, len(user_job_mat.data)))
    user_job_new_mat._shape = (n_users+1, n_jobs)


    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_job_new_mat)

    distances, indices = model.kneighbors(user_job_new_mat[n_users], n_neighbors=n+1)
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1]) [:0:-1]

    reverse_mapper = {v: k for k, v in user_mapper.items()}

    similar_users = []
    for _, (idx, _) in enumerate(raw_recommends):
        similar_users = similar_users + [reverse_mapper[idx]]

    return similar_users


def user_cf_recommend_jobs(user_mapper, job_mapper, selected_jobs, n=10):
    '''returns a list of top n job ids and the user_ids of the nearest neighbours'''
    user_job_mat = load_npz('sparse_user_job.npz')

    nearest_neighbours = similar_users(user_job_mat, user_mapper, job_mapper, selected_jobs)
    recommended_jobs_indices = []

    for i in nearest_neighbours:
        user_index = user_mapper[i]
        user_row = user_job_mat.getrow(user_index)
        (_, nonzero_columns) = user_row.nonzero()
        recommended_jobs_indices = recommended_jobs_indices + nonzero_columns.tolist()

    reverse_job_mapper = {v: k for k, v in job_mapper.items()}

    unique_recommendations = []

    for x in recommended_jobs_indices: 
        if x not in unique_recommendations: 
            unique_recommendations.append(reverse_job_mapper[x]) 
    
    return unique_recommendations[:n], nearest_neighbours


def user_information(users):
    user_info = pd.read_hdf('user-info.h5', 'df')

    user_degrees = pd.Series()  
    user_years_work_experience = pd.Series()
    user_number_managed = pd.Series()    
    
    for user_id in users:
        user_row = user_info[user_info.UserID == user_id]

        user_years_work_experience = user_years_work_experience.append(user_row['TotalYearsExperience'])
        user_number_managed = user_number_managed.append(user_row['ManagedHowMany'])
        user_degrees = user_degrees.append(user_row['DegreeType'])
        
    average_degree = user_degrees.mode().to_string()
    #to remove the number before the string
    average_degree_remove_int = average_degree.split(' ', 1)[1]
    average_degree_formatted = average_degree_remove_int.split()[0]
    if (average_degree_formatted == "None"):
        average_degree_formatted = "no"
    elif (average_degree_formatted == "High"):
        average_degree_formatted = "a High School"
    else:
         average_degree_formatted = "a " + average_degree_formatted

    average_years_experience = user_years_work_experience.mean()
    if not math.isnan(average_years_experience):
        average_years_experience = int(average_years_experience)

    average_number_managed = user_number_managed.mean()
    if not math.isnan(average_number_managed):
        average_number_managed = int(average_number_managed)

    description = ['The algorithm first searched for users who applied to jobs that are similar to the ones you selected, and then generated recommendations based on other jobs these users have applied for. You are similar to users 1, 2 and 3. They have average qualifications of:',
                    html.Br(),
                    html.Ul(children=[
                        html.Li('{} degree'.format(average_degree_formatted)),
                        html.Li('{} years previous work experience'.format(average_years_experience)),
                        html.Li('management experience of > {} other people'.format(average_number_managed) )
                    ])]    

    return description


