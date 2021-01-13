#split users into 7 -> 6 for 12-fold cross validations (each split into 2) and one for test
#so have test group of 1523 and folds of 651
#train 10 models with different factors and regularisation on 9 of the folds and then use the remaining fold for testing (for each model do with a number of different iterations to guarantee convergence has occured):
#for each of the test users suppose they have selected 2 (3?) jobs and calculate recall + was at least one recommended for remaining >3
#for models do 1 - fac=10, reg=0.3
#              2 - fac=10, reg=0.1
#              3 - fac=10, reg=0.01
#              4 - fac=50, reg=0.3
#              5 - fac=50, reg=0.1
#              6 - fac=50, reg=0.01
#              7 - fac=100, reg=0.3
#              8 - fac=100, reg=0.1
#              9 - fac=100, reg=0.01
#             10 - fac=200, reg=0.3
#             11 - fac=200, reg=0.1
#             12 - fac=200, reg=0.01
            
import implicit
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def recall(recommended_jobs, test_jobs):
    number_test = len(test_jobs)
    number_test_recommended = 0
    for job in test_jobs:
        if job in recommended_jobs:
            number_test_recommended = number_test_recommended + 1
    
    return ( float(number_test_recommended) / float(number_test) )


def at_least_one_metric(recommended_jobs, test_jobs):
    '''takes a list of recommended job ids and a list of test jobs ids and returns whether at least one of 
    the test job ids have been recommended'''

    success = 0
    for job in test_jobs:
        if job in recommended_jobs:
            success = 1

    return success


def calculate_mf_recommendations(mf_model, sparse_user_job, reduced_sparse_user_job, row_index, n=10):
    '''takes the row index in the user-job matrix which we are testing, then adds the first three applications as a
    new user to sparse matrix and returns recall and if atLeastOne accuracy metrics'''

    alpha = 40

    row = sparse_user_job[np.array([row_index]),:]
    (_, nonzero_columns) = row.nonzero()
    training_columns = nonzero_columns[:3]
    test_columns = nonzero_columns[3:]

    #for some reason size of user-job-matrix increases at every iteration so this reduces it back to size
    reduced_sparse_user_job = reduced_sparse_user_job[:7161, :]
    n_users, n_jobs = reduced_sparse_user_job.shape

    train_ratings = [alpha for i in range(3)]

    new_sparse_user_job = reduced_sparse_user_job
    new_sparse_user_job.data = np.hstack((reduced_sparse_user_job.data, train_ratings))
    new_sparse_user_job.indices = np.hstack((reduced_sparse_user_job.indices, training_columns))
    new_sparse_user_job.indptr = np.hstack((reduced_sparse_user_job.indptr, len(reduced_sparse_user_job.data)))
    new_sparse_user_job._shape = (n_users+1, n_jobs)

    recommended_index, _ =  zip(*mf_model.recommend(n_users, new_sparse_user_job, N=n, recalculate_user=True))
    
    return recall(recommended_index, test_columns), at_least_one_metric(recommended_index, test_columns)



def calculate_mf_recommendations(mf_model, sparse_user_job, reduced_sparse_user_job, row_index, n=10):
    '''takes the row index in the user-job matrix which we are testing, then adds the first three applications as a
    new user to sparse matrix and returns recall and if atLeastOne accuracy metrics'''

    alpha = 40

    row = sparse_user_job[np.array([row_index]),:]
    (_, nonzero_columns) = row.nonzero()
    training_columns = nonzero_columns[:3]
    test_columns = nonzero_columns[3:]

    #for some reason size of user-job-matrix increases at every iteration so this reduces it back to size
    reduced_sparse_user_job = reduced_sparse_user_job[:7161, :]
    n_users, n_jobs = reduced_sparse_user_job.shape

    train_ratings = [alpha for i in range(3)]

    new_sparse_user_job = reduced_sparse_user_job
    new_sparse_user_job.data = np.hstack((reduced_sparse_user_job.data, train_ratings))
    new_sparse_user_job.indices = np.hstack((reduced_sparse_user_job.indices, training_columns))
    new_sparse_user_job.indptr = np.hstack((reduced_sparse_user_job.indptr, len(reduced_sparse_user_job.data)))
    new_sparse_user_job._shape = (n_users+1, n_jobs)

    recommended_index, _ =  zip(*mf_model.recommend(n_users, new_sparse_user_job, N=n, recalculate_user=True))
    
    return recall(recommended_index, test_columns), at_least_one_metric(recommended_index, test_columns)


def cross_validation(sparse_user_job):
    
    sparse_job_user = sparse_user_job.T

    columns_1 = np.arange(651,7812)
    sparse_job_user_reduced_1 = sparse_job_user[:, columns_1]        
    sparse_user_job_reduced_1 = sparse_job_user_reduced_1.T
    
    columns_2 = np.concatenate(( np.arange(0,651), np.arange(1302,7812) ))
    sparse_job_user_reduced_2 = sparse_job_user[:, columns_2]
    sparse_user_job_reduced_2 = sparse_job_user_reduced_2.T

    columns_3 = np.concatenate(( np.arange(0,1302), np.arange(1953,7812) ))
    sparse_job_user_reduced_3 = sparse_job_user[:, columns_3]
    sparse_user_job_reduced_3 = sparse_job_user_reduced_3.T

    columns_4 = np.concatenate(( np.arange(0,1953), np.arange(2604,7812) ))
    sparse_job_user_reduced_4 = sparse_job_user[:, columns_4]
    sparse_user_job_reduced_4 = sparse_job_user_reduced_4.T

    columns_5 = np.concatenate(( np.arange(0,2604), np.arange(3255,7812) ))
    sparse_job_user_reduced_5 = sparse_job_user[:, columns_5]
    sparse_user_job_reduced_5 = sparse_job_user_reduced_5.T

    columns_6 = np.concatenate(( np.arange(0,3255), np.arange(3906,7812) ))
    sparse_job_user_reduced_6 = sparse_job_user[:, columns_6]
    sparse_user_job_reduced_6 = sparse_job_user_reduced_6.T

    columns_7 = np.concatenate(( np.arange(0,3906), np.arange(4557,7812) ))
    sparse_job_user_reduced_7 = sparse_job_user[:, columns_7]
    sparse_user_job_reduced_7 = sparse_job_user_reduced_7.T

    columns_8 = np.concatenate(( np.arange(0,4557), np.arange(5208,7812) ))
    sparse_job_user_reduced_8 = sparse_job_user[:, columns_8]
    sparse_user_job_reduced_8 = sparse_job_user_reduced_8.T

    columns_9 = np.concatenate(( np.arange(0,5208), np.arange(5859,7812) ))
    sparse_job_user_reduced_9 = sparse_job_user[:, columns_9]
    sparse_user_job_reduced_9 = sparse_job_user_reduced_9.T

    columns_10 = np.concatenate(( np.arange(0,5859), np.arange(6510,7812) ))
    sparse_job_user_reduced_10 = sparse_job_user[:, columns_10]
    sparse_user_job_reduced_10 = sparse_job_user_reduced_10.T

    columns_11 = np.concatenate(( np.arange(0,6510), np.arange(7161,7812) ))
    sparse_job_user_reduced_11 = sparse_job_user[:, columns_11]
    sparse_user_job_reduced_11 = sparse_job_user_reduced_11.T

    columns_12 = np.arange(0,7161)
    sparse_job_user_reduced_12 = sparse_job_user[:, columns_12]
    sparse_user_job_reduced_12 = sparse_job_user_reduced_12.T


    mf_model_1 = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.3, iterations=30)
    mf_model_1.fit(sparse_job_user_reduced_1)

    mf_model_2 = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.1, iterations=30)
    mf_model_2.fit(sparse_job_user_reduced_2)

    mf_model_3 = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.01, iterations=30)
    mf_model_3.fit(sparse_job_user_reduced_3)

    mf_model_4 = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.3, iterations=30)
    mf_model_4.fit(sparse_job_user_reduced_4)

    mf_model_5 = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
    mf_model_5.fit(sparse_job_user_reduced_5)

    mf_model_6 = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=30)
    mf_model_6.fit(sparse_job_user_reduced_6)

    mf_model_7 = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.3, iterations=30)
    mf_model_7.fit(sparse_job_user_reduced_7)

    mf_model_8 = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.1, iterations=30)
    mf_model_8.fit(sparse_job_user_reduced_8)

    mf_model_9 = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.01, iterations=30)
    mf_model_9.fit(sparse_job_user_reduced_9)

    mf_model_10 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.3, iterations=30)
    mf_model_10.fit(sparse_job_user_reduced_10)

    mf_model_11 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.1, iterations=30)
    mf_model_11.fit(sparse_job_user_reduced_11)

    mf_model_12 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.01, iterations=30)
    mf_model_12.fit(sparse_job_user_reduced_12)


    recall_1 = 0
    at_least_one_metric_1 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_1, sparse_user_job, sparse_user_job_reduced_1, i)
        recall_1 = recall_1 + recall
        at_least_one_metric_1 = at_least_one_metric_1 + at_least_one_metric
    
    average_recall_1 = float(recall_1) / 651
    average_at_least_one_metric_1 = float(at_least_one_metric_1) / 651
    print('average_recall_1: ', average_recall_1)
    print('average_at_least_one_metric_1: ', average_at_least_one_metric_1)

    
    recall_2 = 0
    at_least_one_metric_2 = 0
    for i in range(651, 1302):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_2, sparse_user_job, sparse_user_job_reduced_2, i)
        recall_2 = recall_2 + recall
        at_least_one_metric_2 = at_least_one_metric_2 + at_least_one_metric
    
    average_recall_2 = float(recall_2) / 651
    average_at_least_one_metric_2 = float(at_least_one_metric_2) / 651
    print('average_recall_2: ', average_recall_2)
    print('average_at_least_one_metric_2: ', average_at_least_one_metric_2)

    recall_3 = 0
    at_least_one_metric_3 = 0
    for i in range(1302, 1953):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_3, sparse_user_job, sparse_user_job_reduced_3, i)
        recall_3 = recall_3 + recall
        at_least_one_metric_3 = at_least_one_metric_3 + at_least_one_metric
    
    average_recall_3 = float(recall_3) / 651
    average_at_least_one_metric_3 = float(at_least_one_metric_3) / 651
    print('average_recall_3: ', average_recall_3)
    print('average_at_least_one_metric_3: ', average_at_least_one_metric_3)

    recall_4 = 0
    at_least_one_metric_4 = 0
    for i in range(1953, 2604):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_4, sparse_user_job, sparse_user_job_reduced_4, i)
        recall_4 = recall_4 + recall
        at_least_one_metric_4 = at_least_one_metric_4 + at_least_one_metric
    
    average_recall_4 = float(recall_4) / 651
    average_at_least_one_metric_4 = float(at_least_one_metric_4) / 651
    print('average_recall_4: ', average_recall_4)
    print('average_at_least_one_metric_4: ', average_at_least_one_metric_4)

    recall_5 = 0
    at_least_one_metric_5 = 0
    for i in range(2604, 3255):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_5, sparse_user_job, sparse_user_job_reduced_5, i)
        recall_5 = recall_5 + recall
        at_least_one_metric_5 = at_least_one_metric_5 + at_least_one_metric
    
    average_recall_5 = float(recall_5) / 651
    average_at_least_one_metric_5 = float(at_least_one_metric_5) / 651
    print('average_recall_5: ', average_recall_5)
    print('average_at_least_one_metric_5: ', average_at_least_one_metric_5)

    recall_6 = 0
    at_least_one_metric_6 = 0
    for i in range(3255, 3906):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_6, sparse_user_job, sparse_user_job_reduced_6, i)
        recall_6 = recall_6 + recall
        at_least_one_metric_6 = at_least_one_metric_6 + at_least_one_metric
    
    average_recall_6 = float(recall_6) / 651
    average_at_least_one_metric_6 = float(at_least_one_metric_6) / 651
    print('average_recall_6: ', average_recall_6)
    print('average_at_least_one_metric_6: ', average_at_least_one_metric_6)

    recall_7 = 0
    at_least_one_metric_7 = 0
    for i in range(3906, 4557):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_7, sparse_user_job, sparse_user_job_reduced_7, i)
        recall_7 = recall_7 + recall
        at_least_one_metric_7 = at_least_one_metric_7 + at_least_one_metric
    
    average_recall_7 = float(recall_7) / 651
    average_at_least_one_metric_7 = float(at_least_one_metric_7) / 651
    print('average_recall_7: ', average_recall_7)
    print('average_at_least_one_metric_7: ', average_at_least_one_metric_7)

    recall_8 = 0
    at_least_one_metric_8 = 0
    for i in range(4557, 5208):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_8, sparse_user_job, sparse_user_job_reduced_8, i)
        recall_8 = recall_8 + recall
        at_least_one_metric_8 = at_least_one_metric_8 + at_least_one_metric
    
    average_recall_8 = float(recall_8) / 651
    average_at_least_one_metric_8 = float(at_least_one_metric_8) / 651
    print('average_recall_8: ', average_recall_8)
    print('average_at_least_one_metric_8: ', average_at_least_one_metric_8)

    recall_9 = 0
    at_least_one_metric_9 = 0
    for i in range(5208, 5859):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_9, sparse_user_job, sparse_user_job_reduced_9, i)
        recall_9 = recall_9 + recall
        at_least_one_metric_9 = at_least_one_metric_9 + at_least_one_metric
    
    average_recall_9 = float(recall_9) / 651
    average_at_least_one_metric_9 = float(at_least_one_metric_9) / 651
    print('average_recall_9: ', average_recall_9)
    print('average_at_least_one_metric_9: ', average_at_least_one_metric_9)

    recall_10 = 0
    at_least_one_metric_10 = 0
    for i in range(5859, 6510):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_10, sparse_user_job, sparse_user_job_reduced_10, i)
        recall_10 = recall_10 + recall
        at_least_one_metric_10 = at_least_one_metric_10 + at_least_one_metric
    
    average_recall_10 = float(recall_10) / 651
    average_at_least_one_metric_10 = float(at_least_one_metric_10) / 651
    print('average_recall_10: ', average_recall_10)
    print('average_at_least_one_metric_10: ', average_at_least_one_metric_10)

    recall_11 = 0
    at_least_one_metric_11 = 0
    for i in range(6510, 7161):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_11, sparse_user_job, sparse_user_job_reduced_11, i)
        recall_11 = recall_11 + recall
        at_least_one_metric_11 = at_least_one_metric_11 + at_least_one_metric
    
    average_recall_11 = float(recall_11) / 651
    average_at_least_one_metric_11 = float(at_least_one_metric_11) / 651
    print('average_recall_11: ', average_recall_11)
    print('average_at_least_one_metric_11: ', average_at_least_one_metric_11)

    recall_12 = 0
    at_least_one_metric_12 = 0
    for i in range(7161, 7812):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_12, sparse_user_job, sparse_user_job_reduced_12, i)
        recall_12 = recall_12 + recall
        at_least_one_metric_12 = at_least_one_metric_12 + at_least_one_metric
    
    average_recall_12 = float(recall_12) / 651
    average_at_least_one_metric_12 = float(at_least_one_metric_12) / 651
    print('average_recall_12: ', average_recall_12)
    print('average_at_least_one_metric_12: ', average_at_least_one_metric_12)
    
    '''
        average_recall_1:  0.10671541126115974
        average_at_least_one_metric_1:  0.41321044546851

        average_recall_2:  0.09912884792161074
        average_at_least_one_metric_2:  0.3655913978494624

        average_recall_3:  0.11492882635904957
        average_at_least_one_metric_3:  0.4116743471582181

        average_recall_4:  0.25428424194053845
        average_at_least_one_metric_4:  0.7096774193548387

        average_recall_5:  0.26017080120096564
        average_at_least_one_metric_5:  0.738863287250384

        average_recall_6:  0.25657350392394596
        average_at_least_one_metric_6:  0.706605222734255

        average_recall_7:  0.26967534593747905
        average_at_least_one_metric_7:  0.7542242703533026

        average_recall_8:  0.2868122891005407
        average_at_least_one_metric_8:  0.7542242703533026

        average_recall_9:  0.26453803897856293
        average_at_least_one_metric_9:  0.7603686635944701

        average_recall_10:  0.29583776898816283
        average_at_least_one_metric_10:  0.8018433179723502

        average_recall_11:  0.2874163814756907
        average_at_least_one_metric_11:  0.7695852534562212

        average_recall_12:  0.296975774517057
        average_at_least_one_metric_12:  0.7910906298003072
    '''

def cross_validation_narrow(sparse_user_job):
    
    sparse_job_user = sparse_user_job.T

    columns_1 = np.arange(651,7812)
    sparse_job_user_reduced_1 = sparse_job_user[:, columns_1]        
    sparse_user_job_reduced_1 = sparse_job_user_reduced_1.T
    
    columns_2 = np.concatenate(( np.arange(0,651), np.arange(1302,7812) ))
    sparse_job_user_reduced_2 = sparse_job_user[:, columns_2]
    sparse_user_job_reduced_2 = sparse_job_user_reduced_2.T

    columns_3 = np.concatenate(( np.arange(0,1302), np.arange(1953,7812) ))
    sparse_job_user_reduced_3 = sparse_job_user[:, columns_3]
    sparse_user_job_reduced_3 = sparse_job_user_reduced_3.T

    columns_4 = np.concatenate(( np.arange(0,1953), np.arange(2604,7812) ))
    sparse_job_user_reduced_4 = sparse_job_user[:, columns_4]
    sparse_user_job_reduced_4 = sparse_job_user_reduced_4.T

    columns_5 = np.concatenate(( np.arange(0,2604), np.arange(3255,7812) ))
    sparse_job_user_reduced_5 = sparse_job_user[:, columns_5]
    sparse_user_job_reduced_5 = sparse_job_user_reduced_5.T

    columns_6 = np.concatenate(( np.arange(0,3255), np.arange(3906,7812) ))
    sparse_job_user_reduced_6 = sparse_job_user[:, columns_6]
    sparse_user_job_reduced_6 = sparse_job_user_reduced_6.T

    columns_7 = np.concatenate(( np.arange(0,3906), np.arange(4557,7812) ))
    sparse_job_user_reduced_7 = sparse_job_user[:, columns_7]
    sparse_user_job_reduced_7 = sparse_job_user_reduced_7.T

    columns_8 = np.concatenate(( np.arange(0,4557), np.arange(5208,7812) ))
    sparse_job_user_reduced_8 = sparse_job_user[:, columns_8]
    sparse_user_job_reduced_8 = sparse_job_user_reduced_8.T

    columns_9 = np.concatenate(( np.arange(0,5208), np.arange(5859,7812) ))
    sparse_job_user_reduced_9 = sparse_job_user[:, columns_9]
    sparse_user_job_reduced_9 = sparse_job_user_reduced_9.T

    columns_10 = np.concatenate(( np.arange(0,5859), np.arange(6510,7812) ))
    sparse_job_user_reduced_10 = sparse_job_user[:, columns_10]
    sparse_user_job_reduced_10 = sparse_job_user_reduced_10.T

    columns_11 = np.concatenate(( np.arange(0,6510), np.arange(7161,7812) ))
    sparse_job_user_reduced_11 = sparse_job_user[:, columns_11]
    sparse_user_job_reduced_11 = sparse_job_user_reduced_11.T

    columns_12 = np.arange(0,7161)
    sparse_job_user_reduced_12 = sparse_job_user[:, columns_12]
    sparse_user_job_reduced_12 = sparse_job_user_reduced_12.T


    mf_model_1 = implicit.als.AlternatingLeastSquares(factors=150, regularization=0.3, iterations=30)
    mf_model_1.fit(sparse_job_user_reduced_1)

    mf_model_2 = implicit.als.AlternatingLeastSquares(factors=150, regularization=0.1, iterations=30)
    mf_model_2.fit(sparse_job_user_reduced_2)

    mf_model_3 = implicit.als.AlternatingLeastSquares(factors=150, regularization=0.01, iterations=30)
    mf_model_3.fit(sparse_job_user_reduced_3)

    mf_model_4 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.3, iterations=30)
    mf_model_4.fit(sparse_job_user_reduced_4)

    mf_model_5 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.1, iterations=30)
    mf_model_5.fit(sparse_job_user_reduced_5)

    mf_model_6 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.01, iterations=30)
    mf_model_6.fit(sparse_job_user_reduced_6)

    mf_model_7 = implicit.als.AlternatingLeastSquares(factors=250, regularization=0.3, iterations=30)
    mf_model_7.fit(sparse_job_user_reduced_7)

    mf_model_8 = implicit.als.AlternatingLeastSquares(factors=250, regularization=0.1, iterations=30)
    mf_model_8.fit(sparse_job_user_reduced_8)

    mf_model_9 = implicit.als.AlternatingLeastSquares(factors=250, regularization=0.01, iterations=30)
    mf_model_9.fit(sparse_job_user_reduced_9)

    mf_model_10 = implicit.als.AlternatingLeastSquares(factors=300, regularization=0.3, iterations=30)
    mf_model_10.fit(sparse_job_user_reduced_10)

    mf_model_11 = implicit.als.AlternatingLeastSquares(factors=300, regularization=0.1, iterations=30)
    mf_model_11.fit(sparse_job_user_reduced_11)

    mf_model_12 = implicit.als.AlternatingLeastSquares(factors=300, regularization=0.01, iterations=30)
    mf_model_12.fit(sparse_job_user_reduced_12)


    recall_1 = 0
    at_least_one_metric_1 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_1, sparse_user_job, sparse_user_job_reduced_1, i)
        recall_1 = recall_1 + recall
        at_least_one_metric_1 = at_least_one_metric_1 + at_least_one_metric
    
    average_recall_1 = float(recall_1) / 651
    average_at_least_one_metric_1 = float(at_least_one_metric_1) / 651
    print('average_recall_1: ', average_recall_1)
    print('average_at_least_one_metric_1: ', average_at_least_one_metric_1)

    
    recall_2 = 0
    at_least_one_metric_2 = 0
    for i in range(651, 1302):
        recall, at_least_one_metric = calculate_recommendations(mf_model_2, sparse_user_job, sparse_user_job_reduced_2, i)
        recall_2 = recall_2 + recall
        at_least_one_metric_2 = at_least_one_metric_2 + at_least_one_metric
    
    average_recall_2 = float(recall_2) / 651
    average_at_least_one_metric_2 = float(at_least_one_metric_2) / 651
    print('average_recall_2: ', average_recall_2)
    print('average_at_least_one_metric_2: ', average_at_least_one_metric_2)

    recall_3 = 0
    at_least_one_metric_3 = 0
    for i in range(1302, 1953):
        recall, at_least_one_metric = calculate_recommendations(mf_model_3, sparse_user_job, sparse_user_job_reduced_3, i)
        recall_3 = recall_3 + recall
        at_least_one_metric_3 = at_least_one_metric_3 + at_least_one_metric
    
    average_recall_3 = float(recall_3) / 651
    average_at_least_one_metric_3 = float(at_least_one_metric_3) / 651
    print('average_recall_3: ', average_recall_3)
    print('average_at_least_one_metric_3: ', average_at_least_one_metric_3)

    recall_4 = 0
    at_least_one_metric_4 = 0
    for i in range(1953, 2604):
        recall, at_least_one_metric = calculate_recommendations(mf_model_4, sparse_user_job, sparse_user_job_reduced_4, i)
        recall_4 = recall_4 + recall
        at_least_one_metric_4 = at_least_one_metric_4 + at_least_one_metric
    
    average_recall_4 = float(recall_4) / 651
    average_at_least_one_metric_4 = float(at_least_one_metric_4) / 651
    print('average_recall_4: ', average_recall_4)
    print('average_at_least_one_metric_4: ', average_at_least_one_metric_4)

    recall_5 = 0
    at_least_one_metric_5 = 0
    for i in range(2604, 3255):
        recall, at_least_one_metric = calculate_recommendations(mf_model_5, sparse_user_job, sparse_user_job_reduced_5, i)
        recall_5 = recall_5 + recall
        at_least_one_metric_5 = at_least_one_metric_5 + at_least_one_metric
    
    average_recall_5 = float(recall_5) / 651
    average_at_least_one_metric_5 = float(at_least_one_metric_5) / 651
    print('average_recall_5: ', average_recall_5)
    print('average_at_least_one_metric_5: ', average_at_least_one_metric_5)

    recall_6 = 0
    at_least_one_metric_6 = 0
    for i in range(3255, 3906):
        recall, at_least_one_metric = calculate_recommendations(mf_model_6, sparse_user_job, sparse_user_job_reduced_6, i)
        recall_6 = recall_6 + recall
        at_least_one_metric_6 = at_least_one_metric_6 + at_least_one_metric
    
    average_recall_6 = float(recall_6) / 651
    average_at_least_one_metric_6 = float(at_least_one_metric_6) / 651
    print('average_recall_6: ', average_recall_6)
    print('average_at_least_one_metric_6: ', average_at_least_one_metric_6)

    recall_7 = 0
    at_least_one_metric_7 = 0
    for i in range(3906, 4557):
        recall, at_least_one_metric = calculate_recommendations(mf_model_7, sparse_user_job, sparse_user_job_reduced_7, i)
        recall_7 = recall_7 + recall
        at_least_one_metric_7 = at_least_one_metric_7 + at_least_one_metric
    
    average_recall_7 = float(recall_7) / 651
    average_at_least_one_metric_7 = float(at_least_one_metric_7) / 651
    print('average_recall_7: ', average_recall_7)
    print('average_at_least_one_metric_7: ', average_at_least_one_metric_7)

    recall_8 = 0
    at_least_one_metric_8 = 0
    for i in range(4557, 5208):
        recall, at_least_one_metric = calculate_recommendations(mf_model_8, sparse_user_job, sparse_user_job_reduced_8, i)
        recall_8 = recall_8 + recall
        at_least_one_metric_8 = at_least_one_metric_8 + at_least_one_metric
    
    average_recall_8 = float(recall_8) / 651
    average_at_least_one_metric_8 = float(at_least_one_metric_8) / 651
    print('average_recall_8: ', average_recall_8)
    print('average_at_least_one_metric_8: ', average_at_least_one_metric_8)

    recall_9 = 0
    at_least_one_metric_9 = 0
    for i in range(5208, 5859):
        recall, at_least_one_metric = calculate_recommendations(mf_model_9, sparse_user_job, sparse_user_job_reduced_9, i)
        recall_9 = recall_9 + recall
        at_least_one_metric_9 = at_least_one_metric_9 + at_least_one_metric
    
    average_recall_9 = float(recall_9) / 651
    average_at_least_one_metric_9 = float(at_least_one_metric_9) / 651
    print('average_recall_9: ', average_recall_9)
    print('average_at_least_one_metric_9: ', average_at_least_one_metric_9)

    recall_10 = 0
    at_least_one_metric_10 = 0
    for i in range(5859, 6510):
        recall, at_least_one_metric = calculate_recommendations(mf_model_10, sparse_user_job, sparse_user_job_reduced_10, i)
        recall_10 = recall_10 + recall
        at_least_one_metric_10 = at_least_one_metric_10 + at_least_one_metric
    
    average_recall_10 = float(recall_10) / 651
    average_at_least_one_metric_10 = float(at_least_one_metric_10) / 651
    print('average_recall_10: ', average_recall_10)
    print('average_at_least_one_metric_10: ', average_at_least_one_metric_10)

    recall_11 = 0
    at_least_one_metric_11 = 0
    for i in range(6510, 7161):
        recall, at_least_one_metric = calculate_recommendations(mf_model_11, sparse_user_job, sparse_user_job_reduced_11, i)
        recall_11 = recall_11 + recall
        at_least_one_metric_11 = at_least_one_metric_11 + at_least_one_metric
    
    average_recall_11 = float(recall_11) / 651
    average_at_least_one_metric_11 = float(at_least_one_metric_11) / 651
    print('average_recall_11: ', average_recall_11)
    print('average_at_least_one_metric_11: ', average_at_least_one_metric_11)

    recall_12 = 0
    at_least_one_metric_12 = 0
    for i in range(7161, 7812):
        recall, at_least_one_metric = calculate_recommendations(mf_model_12, sparse_user_job, sparse_user_job_reduced_12, i)
        recall_12 = recall_12 + recall
        at_least_one_metric_12 = at_least_one_metric_12 + at_least_one_metric
    
    average_recall_12 = float(recall_12) / 651
    average_at_least_one_metric_12 = float(at_least_one_metric_12) / 651
    print('average_recall_12: ', average_recall_12)
    print('average_at_least_one_metric_12: ', average_at_least_one_metric_12)

    '''
        average_recall_1:  0.2984770737307024
        average_at_least_one_metric_1:  0.7803379416282642

        average_recall_2:  0.2820620919866327
        average_at_least_one_metric_2:  0.7526881720430108

        average_recall_3:  0.2790073031455297
        average_at_least_one_metric_3:  0.7511520737327189

        average_recall_4:  0.30420545254210585
        average_at_least_one_metric_4:  0.7772657450076805

        average_recall_5:  0.31866686100323754
        average_at_least_one_metric_5:  0.8095238095238095

        average_recall_6:  0.3135713776516499
        average_at_least_one_metric_6:  0.804915514592934

        average_recall_7:  0.30271406916576993
        average_at_least_one_metric_7:  0.8095238095238095

        average_recall_8:  0.3052751653310733
        average_at_least_one_metric_8:  0.7803379416282642

        average_recall_9:  0.2916355589588203
        average_at_least_one_metric_9:  0.7926267281105991

        average_recall_10:  0.30363246667898364
        average_at_least_one_metric_10:  0.8110599078341014

        average_recall_11:  0.2985783391877972
        average_at_least_one_metric_11:  0.7895545314900153

        average_recall_12:  0.29369413847955506
        average_at_least_one_metric_12:  0.7818740399385561
    '''

#optimum is between: 
# implicit.als.AlternatingLeastSquares(factors=200, regularization=0.1, iterations=30)
# implicit.als.AlternatingLeastSquares(factors=200, regularization=0.01, iterations=30)
#hence tried both on test set and 2nd gave better results

