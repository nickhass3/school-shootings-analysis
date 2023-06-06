import pandas as pd
import re
from transformers import pipeline

def adjust_for_suicide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # if shooter_killed is True, then subtract 1 from Deaths
    df.loc[df['shooter_killed'] == True, 'Deaths'] = df['Deaths'] - 1

    # if shooter_injured is True, then subtract 1 from Injuries
    df.loc[df['shooter_injured'] == True, 'Injuries'] = df['Injuries'] - 1

    return df

# write a function that runs the model on the description column from the df and creates a new column with the answer
def get_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs an Question Answering model distilbert-base-cased-distilled-squadmodel to get the age of the shooter.
    """
    
    df = df.copy()
    my_question = "How old is the shooter?"
    qa_model = pipeline("question-answering")
    df['age_of_shooter_resp'] = df['Description'].apply(lambda x: qa_model(question = my_question, context = x))
    df['age_of_shooter'] = df['age_of_shooter_resp'].apply(lambda x: x.get('answer') if x.get('score') > 0.4 else None)
    # convert the age_of_shooter column to int
    df['age_of_shooter'] = df['age_of_shooter'].apply(convert_to_int)
    df['age_of_shooter'] = df['age_of_shooter'].astype('Int64')
    # drop the response column
    df = df.drop(columns=['age_of_shooter_resp'])

    return df

def remove_brackets(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Removes all groups of brackets and numbers from the specified column.
    Examples of brackets and numbers:
    before: This is an example [24][29]
    after: This is an example
    """
    # Regular expression pattern to remove all groups of brackets and numbers inside
    pattern = r'\s*\[\d+\]\s*'

    # Remove all groups of brackets and numbers from the specified column
    df[column_name] = df[column_name].apply(lambda text: re.sub(pattern, '', text))

    return df

def convert_to_int(x):
    """
    Tries to converts value to integer, 
    if that value is greater than 100 (no one over the age of 100 shoots people, 
    this is also just because the model might have picked up a number that is not the age), 
    or fails to convert to integer, returns None.
    """
    try:
        return int(x) if int(x) < 100 else None
    except:
        return None

def main():
    df = pd.read_parquet('data/raw/school_shootings.parquet')
    df = adjust_for_suicide(df)
    df = remove_brackets(df, 'Description')
    df['State'] = df['Location'].apply(lambda x: x.split(',')[-1].strip())
    df = get_age(df)
    df.to_parquet('data/clean/school_shootings.parquet', engine='pyarrow')

if __name__ == '__main__':
    main()