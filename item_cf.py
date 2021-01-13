import pandas as pd
import csv
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors
from data_manipulation import fit_cf_model



def similar_jobs(job_user_mat, job_mapper, job_id, n=10):
    '''takes matrix and map from job_id to index in matrix, and job_id and returns the n most similar jobs'''
    model = fit_cf_model(job_user_mat)
    idx = job_mapper[job_id]

    distances, indices = model.kneighbors(job_user_mat[idx], n_neighbors=n+1)
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1]) [:0:-1]

    reverse_mapper = {v: k for k, v in job_mapper.items()}

    similar_jobs = []
    for _, (idx, _) in enumerate(raw_recommends):
        similar_jobs = similar_jobs + [reverse_mapper[idx]]

    return similar_jobs


def job_cf_recommend_jobs(job_mapper, selected_jobs, n=10):
    '''returns a list of top n job_ids'''
    job_user_mat = load_npz('sparse_job_user.npz')

    selected_job_ids = [row["JobID"] for _, row in selected_jobs.iterrows()]

    #list of the lists of recommendations from each job in selected_jobs
    possible_recommendations = []
    for job_id in selected_job_ids:
        possible_recommendations = possible_recommendations + [(job_id, similar_jobs(job_user_mat, job_mapper, job_id))]
    
    unique_recommendations = []
    job_explanation = []
    for i in range(10):
        for (job_id, job_list) in possible_recommendations:
            job = job_list[i]
            if job not in unique_recommendations: 
                unique_recommendations.append(job)
                job_explanation.append(job_id)
    
    return unique_recommendations[:n], job_explanation[:n]


def item_cf_map_jobs(jobIDs, explanationIDs):
    job_info = pd.read_hdf('job-info.h5', 'df')

    explanation_titles = []
    job_data = []

    for i in range(len(jobIDs)):
        job = jobIDs[i]
        explanation = explanationIDs[i]
        job_description = job_info[job_info.JobID == job]
        job_data = job_data + job_description.values.tolist()

        explanation_description = job_info[job_info.JobID == explanation]
        description = explanation_description['Title'].to_string()
        removed_id = description.split(' ', 1)[1] 
        explanation = '''You are recommended this job because you also liked: '%s' which is similar''' %(removed_id)
        explanation_titles = explanation_titles + [explanation]

    recommendations = pd.DataFrame(job_data, columns=['JobID', "Title", "Description", "Requirements", "City", "State", "Country", "Zip"])
    recommendations['Explanations'] = explanation_titles

    return recommendations


