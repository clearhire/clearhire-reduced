import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, load_npz
import implicit
import implicit.evaluation as eval
import pickle
import dash_html_components as html



def model():
    sparse_job_user = load_npz('sparse_job_user.npz')

    model = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.01, iterations=30)
    model.fit(sparse_job_user)

    with open('model.sav', 'wb') as pickle_out:
        pickle.dump(model, pickle_out)



def explain(mf_model, sparse_user_job, user_index, job_id, job_mapper):
    """takes user_index in matrix and job_id as input and returns job_ids which explain a recommendation for a particular user-item pair """
    job_index = job_mapper[job_id]
    
    _, top_contributions, _ = mf_model.explain(user_index, sparse_user_job, job_index, N=2)
    
    reverse_job_mapper = {v: k for k, v in job_mapper.items()}
    top_contributions_ids = []
    for (index, _) in top_contributions:
        top_contributions_ids = top_contributions_ids + [reverse_job_mapper[index]]

    return top_contributions_ids


def mf_recommend_jobs(selected_jobs, job_mapper, n=10):
    '''takes a list of the selected job ids, and a mapper from job_id to column in the user-job matrix
    then adds new user and their applied to jobs to sparse matrix and returns a list of n job_ids'''
    sparse_user_job = load_npz('sparse_user_job.npz')

    with open('model.sav', 'rb') as pickle_in:
        mf_model = pickle.load(pickle_in)

    alpha = 40
    reverse_job_mapper = {v: k for k, v in job_mapper.items()}

    selected_job_ids = [row["JobID"] for _, row in selected_jobs.iterrows()]
    selected_job_indices = [job_mapper[job_id] for job_id in selected_job_ids]

    #for some reason size of user-job-matrix increases at every iteration so this reduces it back to size
    sparse_user_job = sparse_user_job[:9123, :]
    n_users, n_jobs = sparse_user_job.shape

    ratings = [alpha for i in range(len(selected_job_indices))]

    new_sparse_user_job = sparse_user_job
    new_sparse_user_job.data = np.hstack((sparse_user_job.data, ratings))
    new_sparse_user_job.indices = np.hstack((sparse_user_job.indices, selected_job_indices))
    new_sparse_user_job.indptr = np.hstack((sparse_user_job.indptr, len(sparse_user_job.data)))
    new_sparse_user_job._shape = (n_users+1, n_jobs)

    recommended_index, _ =  zip(*mf_model.recommend(n_users, new_sparse_user_job, N=n, recalculate_user=True))
    
    explanations = []
    recommended = []
    for index in recommended_index:
        job_id = reverse_job_mapper[index]
        recommended = recommended + [job_id]
        explanations = explanations + [explain(mf_model, new_sparse_user_job, n_users, job_id, job_mapper)]

    return recommended, explanations


def mf_map_jobs(jobIDs, explanationIDs):
    job_info = pd.read_hdf('job-info.h5', 'df')

    explanations = []
    job_data = []

    for i in range(len(jobIDs)):
        explanation_titles = []
        job = jobIDs[i]
        explanation_ids = explanationIDs[i]

        job_description = job_info[job_info.JobID == job]
        job_data = job_data + job_description.values.tolist()

        for j in explanation_ids:
            explanation_description = job_info[job_info.JobID == j]
            description = explanation_description['Title'].to_string()
            #need to remove first word from string which is job_id so we are just left with job title 
            removed_id = description.split(' ', 1)[1] 
            explanation_titles = explanation_titles + [removed_id]
        
        if (len(explanation_titles) > 1):
            explanation = '''You are recommended this job because you selected: '%s' and '%s'
                            ''' %(explanation_titles[0], explanation_titles[1])

        elif (len(explanation_titles) > 0):
            explanation = '''You are recommended this job because you selected: '%s'
                            ''' %(explanation_titles[0])
        else:
            explanation = ''' '''

        explanations = explanations + [explanation]

    recommendations = pd.DataFrame(job_data, columns=['JobID', "Title", "Description", "Requirements", "City", "State", "Country", "Zip"])
    recommendations['Explanations'] = explanations

    return recommendations

