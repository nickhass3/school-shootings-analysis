import pandas as pd

def create_state_shootings():
    df = pd.read_parquet('data/clean/school_shootings.parquet')

    # create df of number of shootings by state
    state_shootings = df['State'].value_counts().reset_index()
    state_shootings.columns = ['State', 'Number of Shootings']

    state_population = pd.read_csv('data/state-populations.csv')

    state_shootings = state_shootings.merge(state_population, on='State')

    state_shootings['Shootings per 100k'] = state_shootings['Number of Shootings'] / state_shootings['2018 Population'] * 100000

    state_shootings.to_csv('data/state_shootings.csv', index=False)

def main():
    create_state_shootings()


if __name__ == '__main__':
    main()