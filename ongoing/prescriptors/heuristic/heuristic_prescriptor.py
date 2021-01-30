import numpy as np
import pandas as pd
import os,sys

from ongoing.prescriptors.base import BasePrescriptor, NPI_COLUMNS, NPI_MAX_VALUES
import ongoing.prescriptors.base as base

NUM_PRESCRIPTIONS = 10

class Heuristic(BasePrescriptor):
    def __init__(self, seed=base.SEED):
        super().__init__(seed=seed)

    def fit(self, hist_df):
        # there's nothing to be learned in this model, so just return
        return

    def prescribe(self,
                  start_date_str,
                  end_date_str,
                  prior_ips_df,
                  cost_df):

        # Generate prescriptions
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        prescription_dict = {
            'PrescriptionIndex': [],
            'CountryName': [],
            'RegionName': [],
            'Date': []
        }
        for ip in NPI_COLUMNS:
            prescription_dict[ip] = []

        for country_name in prior_ips_df['CountryName'].unique():
            country_df = prior_ips_df[prior_ips_df['CountryName'] == country_name]
            for region_name in country_df['RegionName'].unique():
                geoid = (country_name if isinstance(region_name, float)  # if region_name is Nan, it is float; otherwise it's str
                         else country_name + ' / ' + region_name)

                # Sort IPs for this geo by weight
                geo_costs = cost_df[cost_df['GeoID'] == geoid][NPI_COLUMNS]
                ip_names = list(geo_costs.columns)
                ip_weights = geo_costs.values[0]
                ip_weights = [e/sum(ip_weights) for e in ip_weights]

#                 sorted_ips = [ip for _, ip in sorted(zip(ip_weights, ip_names))]

                # Initialize the IPs to all turned off
#                 curr_ips = {ip: 0 for ip in NPI_MAX_VALUES}



                for prescription_idx in range(NUM_PRESCRIPTIONS):
                    P =  (prescription_idx + 1)*5
                    P = np.percentile(np.array(ip_weights), P)

                    # Turn on the next IP
#                     next_ip = sorted_ips[prescription_idx]
#                     curr_ips[next_ip] = NPI_MAX_VALUES[next_ip]

                    # Use curr_ips for all dates for this prescription
                    for date in pd.date_range(start_date, end_date):
                        prescription_dict['PrescriptionIndex'].append(prescription_idx)
                        prescription_dict['CountryName'].append(country_name)
                        prescription_dict['RegionName'].append(region_name)
                        prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
#                         for ip in NPI_COLUMNS:
#                             prescription_dict[ip].append(curr_ips[ip])
                        for ii in range(len(NPI_COLUMNS)):
                            v = 0  
                            if (ip_weights[ii] <= P):
                                v = NPI_MAX_VALUES[NPI_COLUMNS[ii]]
                            prescription_dict[NPI_COLUMNS[ii]].append(v)                            

        # Create dataframe from dictionary.
        prescription_df = pd.DataFrame(prescription_dict)

        return prescription_df

