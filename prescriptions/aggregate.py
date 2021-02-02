import pandas as pd
import numpy as np

# number of checkpoints
num_required = 10

def get_non_dom_ind(vals_arr, verbose=True):
    res_arr = []
    for i in range(0, len(vals_arr)):
        point1 = vals_arr[i]
        is_dominated = False
        for j in range(0, len(vals_arr)):
            if i == j:
                continue
            point2 = vals_arr[j]
            if ((point2[0] < point1[0]) and (point2[1] <= point1[1])) or (
                    (point2[0] <= point1[0]) and (point2[1] < point1[1])):
                if verbose:
                    print('{} is dominated by {}'.format(i, j))
                is_dominated = True
                break
            pass
        if not is_dominated:
            res_arr.append(i)
        pass
    return res_arr


def get_best_n_points_no_curve(n, arr_input):
    arr_list = arr_input.tolist()
    arr_list.sort(reverse=True, key=lambda x: x[0])
    arr_list = np.array(arr_list)

    # 2. find arc length
    arc_len_arr = []
    for pos in range(0, len(arr_input) - 1):
        p1 = np.array([arr_list[pos][0], arr_list[pos][1]])
        p2 = np.array([arr_list[pos + 1][0], arr_list[pos + 1][1]])
        arc_len_arr.append(np.linalg.norm(p2 - p1))
    arc_len_arr = np.array(arc_len_arr)
    # distance delta
    d = sum(arc_len_arr) / (n - 1)
    # cumul_sum of art length
    arc_len_arr_cum = np.cumsum(arc_len_arr)

    # 3. choose ref. points
    # positions of reference points
    points_pos = [0]
    j = 1
    for i in range(0, len(arc_len_arr_cum)):
        if arc_len_arr_cum[i] >= j * d:
            points_pos.append(i + 1)
            j += 1
            if j == n - 1:
                break
        pass
    points_pos.append(len(arr_list) - 1)

    chosen_points = []
    for ref_point_pos in points_pos:
        ref_point = arr_list[ref_point_pos]
        dist = np.linalg.norm((arr_input - ref_point), axis=1)
        pos = np.argmin(dist)
        chosen_points.append(pos)
        pass

    return chosen_points


def aggregate_results(results):
    df_arr = []
    for el in results:
        df = pd.read_csv(el)
        df.name = el
        df_arr.append(df)

    # get all test
    all_tests = []
    for df in df_arr:
        all_tests.extend(df['TestName'].unique())
        pass
    # remove duplicates
    all_tests = list(set(all_tests))

    # test = all_tests[0]

    final_dfs = []
    for test in all_tests:
        print('Processing {}'.format(test))
        # generate sub_dfs that will contain only the results for the required test
        test_df_arr = []
        for df in df_arr:
            test_df = df.loc[df['TestName'] == test].copy()
            test_df.name = df.name
            test_df['Geo'] = test_df['CountryName'].astype(str) + '__' + test_df['RegionName'].astype(str)
            test_df_arr.append(test_df)
            pass

        # get all geo-s in this test
        all_geos = set()
        for test_df in test_df_arr:
            geos = test_df['Geo'].unique()
            all_geos.update(set(geos))
            pass
        all_geos = list(all_geos)

        # for every geo
        for curr_geo in all_geos:
            # put all prescriptors for the current geo in the same df
            prescr_df = []
            for test_df in test_df_arr:
                df = test_df.loc[test_df['Geo'] == curr_geo].copy()
                df['source'] = test_df.name
                prescr_df.append(df)
                pass
            # put them all together and remove duplicates
            prescr_df = pd.concat(prescr_df).drop_duplicates(['PredictedDailyNewCases', 'Stringency'])

            # choose non-dominated
            arr = prescr_df[['PredictedDailyNewCases', 'Stringency']].values
            non_dom_idx = get_non_dom_ind(arr, verbose=False)

            # check here if we have at least 10
            if len(non_dom_idx) < num_required:
                # print('less than 10: {}'.format(curr_geo))
                # just copy the first point the required number of times
                chosen_points_pos = non_dom_idx
                num_found = len(chosen_points_pos)
                for i in range(0, num_required - num_found):
                    chosen_points_pos.append(chosen_points_pos[0])
                    pass
                chosen_non_dom = chosen_points_pos
                pass
            elif len(non_dom_idx) == num_required:
                # don't do anything
                chosen_points_pos = non_dom_idx
                chosen_non_dom = chosen_points_pos
                pass
            else:
                # choose points
                chosen_points_pos = get_best_n_points_no_curve(num_required, arr[non_dom_idx])
                chosen_non_dom = np.array(non_dom_idx)[chosen_points_pos]
                pass
            # create a df with chosen tests for this geo
            geo_df = prescr_df.iloc[chosen_non_dom].copy()
            geo_df.drop('Unnamed: 0', axis=1, inplace=True)
            tmp = geo_df.copy()
            while len(geo_df) < num_required:
                geo_df = pd.concat([geo_df, tmp])
            geo_df = geo_df[0: num_required]
            geo_df['source-PrescriptionIndex'] = geo_df['PrescriptionIndex']
            geo_df['PrescriptionIndex'] = [i for i in range(0, num_required)]
            final_dfs.append(geo_df)
            pass
        pass
    res_df = pd.concat(final_dfs)
    res_df.reset_index(drop=True, inplace=True)
    return res_df

# path to files with results
results = [
        'Heuristic_pct_5.csv',
        'neat_evaluate_2D_big_test.csv',
    ]
final_df = aggregate_results(results)

final_df.to_csv('aggregation.csv')
