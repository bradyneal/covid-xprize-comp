import numpy as np
import pandas as pd
import os

from ongoing.prescriptors.base import BasePrescriptor
import ongoing.prescriptors.base as base

NUM_PRESCRIPTIONS = 10

IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}

IP_COLS = list(IP_MAX_VALUES.keys())


class BlindGreedy(BasePrescriptor):
    def __init__(self, seed=base.SEED):
        super().__init__(seed=seed)

    def fit(self, data):
        # there's nothing to be learned in this model, so just return
        return

    def prescribe(self,
                  start_date_str,
                  end_date_str,
                  prior_ips,
                  costs):

        # Generate prescriptions
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        prescription_dict = {
            'PrescriptionIndex': [],
            'CountryName': [],
            'RegionName': [],
            'Date': []
        }
        for ip in IP_COLS:
            prescription_dict[ip] = []

        for country_name in prior_ips['CountryName'].unique():
            country_df = prior_ips[prior_ips['CountryName'] == country_name]
            for region_name in country_df['RegionName'].unique():
                geoid = (country_name if isinstance(region_name, float)  # if region_name is Nan, it is float; otherwise it's str
                         else country_name + ' / ' + region_name)

                # Sort IPs for this geo by weight
                geo_costs = costs[costs['GeoID'] == geoid][IP_COLS]
                ip_names = list(geo_costs.columns)
                ip_weights = geo_costs.values[0]
                sorted_ips = [ip for _, ip in sorted(zip(ip_weights, ip_names))]

                # Initialize the IPs to all turned off
                curr_ips = {ip: 0 for ip in IP_MAX_VALUES}

                for prescription_idx in range(NUM_PRESCRIPTIONS):

                    # Turn on the next IP
                    next_ip = sorted_ips[prescription_idx]
                    curr_ips[next_ip] = IP_MAX_VALUES[next_ip]

                    # Use curr_ips for all dates for this prescription
                    for date in pd.date_range(start_date, end_date):
                        prescription_dict['PrescriptionIndex'].append(prescription_idx)
                        prescription_dict['CountryName'].append(country_name)
                        prescription_dict['RegionName'].append(region_name)
                        prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
                        for ip in IP_COLS:
                            prescription_dict[ip].append(curr_ips[ip])

        # Create dataframe from dictionary.
        prescription_df = pd.DataFrame(prescription_dict)

        return prescription_df


if __name__ == '__main__':
    prescriptor = BlindGreedy(seed=42)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, 'prescriptions')
    ofile_path = os.path.abspath(os.path.join(output_dir, 'blind_greedy_evaluate2.csv'))
    print(ofile_path)
    print()
    prescriptor.evaluate(output_file_path=ofile_path)
