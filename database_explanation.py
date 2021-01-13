import pandas as pd
import math
from scipy.sparse import load_npz
from data_manipulation import load_data


def calculate_averages(job_mapper, user_mapper):
    '''creates a pandas dataframe with jobs as index and averages as columns'''
    sparse_job_user = load_npz('sparse_job_user.npz')
    job_info = pd.read_hdf('job-info.h5', 'df')
    user_info = pd.read_hdf('user-info.h5', 'df')

    reverse_mapper = {v: k for k, v in user_mapper.items()}

    user_stats_dict = {}
    for _, row in job_info.iterrows():
        #should I work these out by taking the mean or mode --> currently using the mode
        user_degrees = pd.Series()
        user_years_work_experience = pd.Series()
        user_number_managed = pd.Series()

        job_id = row['JobID']
        job_index = job_mapper[job_id]
        mat_row = sparse_job_user[job_index]
        (_, user_indices) = mat_row.nonzero()

        for index in user_indices:
            user_id = reverse_mapper[index]
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

        user_stats_dict[job_id] = ['''The average qualifications of previous applicants include: {0} degree, {1} years previous work experience,
                                    and experience of managing > {2} people
                                    '''.format(average_degree_formatted, average_years_experience, average_number_managed)]
    
    df = pd.DataFrame.from_dict(user_stats_dict, orient='index', columns=['Explanation'])
    
    df.to_hdf('database-explanation.h5', key='df', mode='w')
    return(df)
    

def db_explanation_map_jobs(jobIDs):
    job_info = pd.read_hdf('job-info.h5', 'df')
    db_explanation_df = pd.read_hdf('database-explanation.h5', 'df')

    explanations = []
    job_data = []

    for i in range(len(jobIDs)):
        job = jobIDs[i]
        explanations = explanations + [db_explanation_df.loc[job, 'Explanation']]
        job_description = job_info[job_info.JobID == job]
        job_data = job_data + job_description.values.tolist()

    recommendations = pd.DataFrame(job_data, columns=['JobID', "Title", "Description", "Requirements", "City", "State", "Country", "Zip"])
    recommendations['Explanations'] = explanations

    return recommendations

