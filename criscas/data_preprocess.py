import pandas as pd
import numpy as np
from scipy import stats

# function to extend each concatenated sequence into its bases
def get_char(seq):
    """split string int sequence of chars returned in pandas.Series"""
    chars = list(seq)
    return pd.Series(chars)
    
def process_perbase_df(df, target_base):
    """cleans a data frame representing sequences and their edit info obtained from crispr experiment
    
    Args:
        df: pandas.DataFrame
        cutoff_score: float, threshold for creating two categories from edit score
    
    Note:
        assumed columns in the dataframe are: [ID, seq, allCounts, V1, V2, ..., V20]
    """
    target_cols = ['ID', 'seq']
    df = df[target_cols].copy()
    # harmonize sequence string representation to capitalized form
    df['seq'] = df['seq'].str.upper()
    # allocate sequences to train vs. val/test
    df['seq_type'] = 1

    baseseq_df = df['seq'].apply(get_char)
    baseseq_df.columns = [f'B{i}' for  i in range(1, 21)]
    base_mask = (baseseq_df == target_base) * 1
    base_mask.columns = [f'M{i}' for  i in range(1, 21)]
    
    baseseq_letters_df = baseseq_df.copy()
    baseseq_letters_df.columns = [f'L{i}' for  i in range(1, 21)]
    # replace base letters with numbers
    baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
    base_df = pd.concat([base_mask,
                         df,
                         baseseq_letters_df,
                         baseseq_df], axis=1)
    # keep sequences that have target base
    # M1..M20 is mask for target base occurrence
    base_df = base_df[(base_df[[f'M{i}' for  i in range(1, 21)]].sum(axis=1) != 0)].copy()
    base_df.reset_index(inplace=True)


    return base_df

def validate_df(df):
    mask_cols = [f'M{i}' for  i in range(1, 21)]
    print('number of NA:', df.isna().any().sum())
    print('number of sequences without target base:', (df[mask_cols].sum(axis=1) == 0).sum())

def compute_rank(s, eff_scores=None):
    return stats.percentileofscore(eff_scores, s, kind='weak')