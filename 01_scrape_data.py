import pandas as pd

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    for index, row in df.iterrows():
        # if all values in the row are numbers
        if row.str.isnumeric().all():
            # drop that row
            df.drop(index, inplace=True)
    
    df.columns = ['Date', 'Location', 'Deaths', 'Injuries', 'Total', 'Description']
    
    return df

def remove_footnotes(df: pd.DataFrame) -> pd.DataFrame:
    # if df['Deaths'] has [n 1], then create another column called shooter_killed and set it equal to True
    df['shooter_killed'] = df['Deaths'].str.contains('\[n 1\]')

    # remove the [n 1] from the Deaths column
    df['Deaths'] = df['Deaths'].str.replace('\[n 1\]', '')

    # if df['Injuries'] has [n 2], then create another column called shooter_injured and set it equal to True
    df['shooter_injured'] = df['Injuries'].str.contains('\[n 1\]')

    # remove the [n 2] from the Injuries column
    df['Injuries'] = df['Injuries'].str.replace('\[n 1\]', '')

    # remove anything anything after [n 3] in the Deaths column
    df['Deaths'] = df['Deaths'].astype(str).str.replace('\[n.*', '')
    df['Injuries'] = df['Injuries'].astype(str).str.replace('\[n.*', '')
    df['Total'] = df['Total'].astype(str).str.replace('\[n.*', '')

    # remove anything after 3[ in the Deaths column
    df['Deaths'] = df['Deaths'].astype(str).str.replace('\[.*', '')
    df['Injuries'] = df['Injuries'].astype(str).str.replace('\[.*', '')
    df['Total'] = df['Total'].astype(str).str.replace('\[.*', '')

    return df

def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Date'] = pd.to_datetime(df['Date'])
    df['Location'] = df['Location'].astype(str)
    df['Deaths'] = df['Deaths'].astype(int)
    df['Injuries'] = df['Injuries'].astype(int)
    df['Total'] = df['Total'].astype(int)
    df['Description'] = df['Description'].astype(str)
    
    return df

def main():
    url = 'https://en.wikipedia.org/wiki/List_of_school_shootings_in_the_United_States_(2000%E2%80%93present)'
    tables = pd.read_html(url)

    tables_to_keep = [0, 1, 3]
    for table in tables_to_keep:
        tables[table] = clean_table(tables[table])

    # concatenate the tables we want to keep
    df = pd.concat([tables[0], tables[1], tables[3]], ignore_index=True)
    df.drop(df[df['Location'].str.isnumeric()].index, inplace=True)

    df = remove_footnotes(df)

    df = set_dtypes(df)

    # write df to a parquet file
    df.to_parquet('data/raw/school_shootings.parquet', engine='pyarrow')

if __name__ == '__main__':
    main()