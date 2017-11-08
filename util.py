import numpy as np
import pandas as pd

def get_split_cols(df):
    numerics = []
    categorical = []

    for col in df:
        if((df[col].dtype == np.float64 or df[col].dtype == np.int64) and col != 'Unnamed: 0' \
           and col != 'Electricity' and col != 'Agricultural Household indicator'):
            numerics.append(col)

        else:
            categorical.append(col)
            
    return numerics, categorical
    


def get_split_frame(df):
    '''
        Returns numeric, categorical dfs as a tuple
    '''
    numerics, categorical = get_split_cols(df)

    categorical_df = df[categorical]
    numeric_df = df[numerics]
    
    return numeric_df, categorical_df

def cosine_sim(u,v):
    res = np.array(np.dot(u, v.T) / (np.sqrt(np.dot(u,u.T)) * np.sqrt(np.dot(v,v.T)))).item(0,0)
    
    if not np.isnan(res):
        return res
    
    else:
        return 1e-5
    
def clean_jobs(df):
    jobs = np.unique(list(df['Household Head Occupation']))
    
    newJobs = np.array([])
    for job in jobs:
        temp = job.split(' ')
        temp = [word.lower() for word in temp if word.lower() not in stopwords.words('english')]
        newJobs = np.append(newJobs," ".join(temp))
        
    return newJobs

def stem_jobs(df):
    ps = PorterStemmer()
    newJobs = clean_jobs(df)
    stemmed_jobs = np.array([])
    
    for job in newJobs:
        temp = job.split(' ')
        temp = [ps.stem(word) for word in temp]
        stemmed_jobs = np.append(stemmed_jobs," ".join(temp))
        
    return stemmed_jobs

def cluster_jobs(df):
    stemmed_jobs = stem_jobs(df)
    numJobs = len(np.unique(df['Household Head Occupation']))
    
    # convert jobs to tf-idf vectors
    tfidf_vectorizer=TfidfVectorizer()
    tfidf_matrix=tfidf_vectorizer.fit_transform(stemmed_jobs)
    
    # get cosine similiarity between each job
    cos_sims = np.array([])
    for i in range(np.size(stemmed_jobs)):
        for j in range(np.size(stemmed_jobs)):
            cos_sims = np.append(cos_sims, cosine_sim(tfidf_matrix[i],tfidf_matrix[j]))
            
    cos_sims = np.reshape(cos_sims, (numJobs, numJobs))
    
    # cluster jobs into groupings
    mat = np.matrix(cos_sims)
    groups = SpectralClustering(30).fit_predict(mat)

    # add column of job category
    jobs = np.unique(list(df['Household Head Occupation']))
    jobIndexes = np.concatenate([np.where(jobs == job)[0] for job in df['Household Head Occupation']])
    jobCategory = np.array([str(groups[num]) for num in jobIndexes])
    
    return jobCategory

def get_income_zscores(df):
    regions = pd.DataFrame(df[['Total Household Income', 'Region']]
        .groupby('Region').agg([np.mean, np.std]).to_records())
    regions.columns = ['Region', 'Average Household Income', 'Standard Deviation']

    z_scores = []

    for income, region in zip(df['Total Household Income'], df['Region']):
        idx = list(regions['Region']).index(region)
        avg = list(regions['Average Household Income'])[idx]
        sd = list(regions['Standard Deviation'])[idx]

        z_scores.append((income - avg) / sd)

    return z_scores

def group_highest_grade(df):
    grades = []
    
    for grade in df['Household Head Highest Grade Completed']:
        grade = grade.lower().strip()

        if 'programs' in grade:
            grades.append('Program/Trade School')

        elif 'preschool' in grade:
            grades.append('Preschool')

        elif 'grade' in grade or 'elementary' in grade:
            grades.append('Elementary')

        elif 'high school' in grade:
            grades.append('High School')

        elif 'college' in grade:
            grades.append('College')

        elif 'secondary' in grade:
            grades.append('Post Secondary')

        elif 'baccalaureate' in grade:
            grades.append('Baccalaureate')
            
    return grades

def percentage_employed(df):
    df['Percentage of Family Employed'] = df['Total number of family members employed'] / df['Total Number of Family members']
    df['Interaction between family members and employed'] = df['Total number of family members employed'] * df['Total Number of Family members']
    
    