import os
import pandas as pd

# Sample the same number of accreted stars from each progenitor event

features = ["E","L","FeH","aFe","x","y","z","vx","vy","vz","r","age"]
prog_properties = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']

# Initialise empty arrays to store the progenitor properties
df_list = []
N_progenitors = 0

data_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observations/data/"
galaxy_files = os.listdir(data_dir)

for file in galaxy_files:

    # Load dataframe
    df = pd.read_pickle(f"{data_dir}{file}")

    # Determine the number of progenitors
    progenitor_ids = df['progID'].unique()
    N_progenitors += len(progenitor_ids)

    print(f"N progenitors: {len(progenitor_ids)}", flush=True)

    for progID in progenitor_ids:

        # Get progenitor data
        progenitor_df = df[df['progID'] == progID]

        # Decide how many stars to sample from each progenitor
        n_stars_per_progenitor = min(10000, len(progenitor_df))

        # Sample stars from progenitor
        df_list.append(progenitor_df.sample(n=n_stars_per_progenitor, random_state=42)[features+prog_properties+['progID']])

# Concatenate all progenitor dataframes
df = pd.concat(df_list, ignore_index=True)
print(f"Total number of progenitors: {N_progenitors:,}", flush=True)
print(f"Total number of stars: {len(df):,}", flush=True)
# Save the data
df.to_pickle('/mnt/aridata1/users/ariasant/MW-sbi/data/auriga_sbi_data.pkl')




