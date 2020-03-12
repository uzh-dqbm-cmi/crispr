import pandas as pd


# function to extend each concatenated sequence into its bases
def get_char(seq):
    """split string int sequence of chars returned in pandas.Series"""
    chars = list(seq)
    return pd.Series(chars)

def generate_clean_df(df):
    """cleans a data frame representing sequences and their edit info obtained from crispr experiment
    
    Args:
        df: pandas.DataFrame
    
    Note:
        assumed columsn in the dataframe are: [ID, seq, allCounts, V1, V2, ..., V20]
    """
    # main target columns recording edit per base info
    editbase_cols = [f'V{i}' for i in range(1,21)]
    target_cols = ['ID', 'seq', 'allCounts'] + editbase_cols
    df = df[target_cols].copy()
    # remove NaN sequences
    df = df[df['allCounts'].notna()].copy()
    # harmonize sequence string representation to capitalized form
    df['seq'] = df['seq'].str.upper()
    # compute efficiency score
    df['efficiency_score'] = df[editbase_cols].sum(axis=1)/df['allCounts']
    # create two-classes of sequences (sequences with 0 edits vs. edited ones)
    df['edited_seq_categ'] = 0
    df.loc[df['efficiency_score'] > 0, 'edited_seq_categ'] = 1
    # allocate sequences to train vs. val/test
    df['seq_type'] = 1
    df.loc[df['ID'].str.startswith("RANDOM"), 'seq_type'] = 0
    return df

def generate_perbase_df(df, target_base):
    """generate data frame with efficiency score per base
    
    Args:
        df: pandas.DataFrame (generated from :func:`generate_clean_df`)
        target_base: string, targe base to be edited (i.e. 'A' or 'C')
    
    """
    # main target columns recording edit per base info
    editbase_cols = [f'V{i}' for i in range(1,21)]
    baseseq_df = df['seq'].apply(get_char)
    baseseq_df.columns = [f'B{i}' for  i in range(1, 21)]
    base_mask = (baseseq_df == target_base) * 1
    base_mask.columns = [f'M{i}' for  i in range(1, 21)]
    
    baseseq_letters_df = baseseq_df.copy()
    baseseq_letters_df.columns = [f'L{i}' for  i in range(1, 21)]
    # replace base letters with numbers
    baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
    base_df = pd.concat([base_mask,
                         df[['ID', 'seq_type', 'efficiency_score', 'edited_seq_categ'] + editbase_cols],
                         baseseq_letters_df,
                         baseseq_df], axis=1)
    return base_df
    