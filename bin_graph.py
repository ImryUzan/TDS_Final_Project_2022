
# bin_graph params:
#         required:
#             n_clusters (int)
#             X_axi (string)
#             Y_axi (string)
#             dtf
#         possible:
#             hight_y (int) difult = 4
#             width_x (int)  difult = 10
#             test_mode (BOOL) difult = False

#if test_mode = False , the function bin_graph will show only the best graph for the number of bins that was given by the user
#if test mode = True ,  the function also print the average distance from the graph, of validation set that was generated inside the function (0.1 of the data).

import numpy as np
import sklearn
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from numpy.core.numeric import NaN
from sklearn.utils import shuffle


def cutNumbersForGraph(orginialList):
  temp = orginialList
  for kk in range(0,len(temp)):
    val0 = float('%.3f'%(temp[kk][0]))
    val1 = float('%.3f'%(temp[kk][1]))
    temp[kk] = (val0,val1)
  return temp





# parameters:  n_clusters(int), X_axi(string), Y_axi(string), dtf(DataFrame)
def basicBinsGraph(n_clusters, X_axi, Y_axi, dtf, hight=4, width=10, is_test_mode=False, testset=None, theards=0.3,num_std=1.5,
              should_print=False):
    km = KMeans(n_clusters=n_clusters)
    y_predicted = km.fit_predict(dtf[[X_axi]])
    dtf['cluster'] = y_predicted
    list_float_cluster_centers = []
    for i in km.cluster_centers_:
        list_float_cluster_centers.append(i[0])
    sorted_cluster_centers_list = []
    list_float_cluster_centers.sort()

    clasters_list = []
    for j in range(0, n_clusters):
        clasters_list.append(dtf[dtf.cluster == j])

    clasters_list_sorted_mean = []
    dict_mean_clasterd = {}

    no_outlay_list = []
    for bin_num in range(0, len(clasters_list)):
        mean = clasters_list[bin_num][Y_axi].describe()['mean']
        std = clasters_list[bin_num][Y_axi].describe()['std']
        mean_x = clasters_list[bin_num][X_axi].describe()['mean']
        clasters_list_sorted_mean.append(mean_x)
        dict_mean_clasterd[mean_x] = clasters_list[bin_num]
        max1 = mean + num_std * std
        min1 = mean - num_std * std
        no_outlay = (clasters_list[bin_num][clasters_list[bin_num][Y_axi]< max1])
        no_outlay = (no_outlay[no_outlay[Y_axi]> min1])
        no_outlay_list.append(no_outlay)


    clasters_list_sorted = []
    clasters_list_sorted_mean.sort()
    for mean in clasters_list_sorted_mean:
        clasters_list_sorted.append(dict_mean_clasterd[mean])

    clasters_list = clasters_list_sorted
    result_no_outlay = pd.concat(no_outlay_list)
    if should_print:
        fig, ax = plt.subplots(figsize=(width, hight))

    lisT_save_tupple_years = []
    is_first = True
    min = 0
    for i in dtf[X_axi]:
        if is_first:
            is_first = False
            min = i
        else:
            if i < min:
                min = i

    last_time = min

    for i in list_float_cluster_centers:
        lisT_save_tupple_years.append((last_time, i))
        last_time = i + 0.000001

    for bin_num in range(0, len(lisT_save_tupple_years) - 1):
        first_empty = False
        sec_empty = False
        high = lisT_save_tupple_years[bin_num][1]
        low = lisT_save_tupple_years[bin_num][0]
        range_of = high - low
        range_of_div = range_of / 3
        fir = low + range_of_div
        sec = fir + range_of_div

        low_part = clasters_list[bin_num][clasters_list[bin_num][X_axi] < fir]
        high_part = clasters_list[bin_num][clasters_list[bin_num][X_axi] >= fir]
        if len(low_part) == 0 or len(high_part) == 0:
            first_empty = True

        low_part_2 = clasters_list[bin_num][clasters_list[bin_num][X_axi] < sec]
        high_part_2 = clasters_list[bin_num][clasters_list[bin_num][X_axi] >= sec]

        if len(low_part_2) == 0 or len(high_part_2) == 0:
            sec_empty = True

        if len(low_part) / len(clasters_list[bin_num]) < theards or len(high_part) / len(
                clasters_list[bin_num]) < theards:
            first_empty = True

        if len(low_part_2) / len(clasters_list[bin_num]) < theards or len(high_part_2) / len(
                clasters_list[bin_num]) < theards:
            sec_empty = True

        if (first_empty == False and sec_empty == True):
            low_part = low_part[Y_axi]
            high_part = high_part[Y_axi]
            mean_low = low_part.mean()
            mean_high = high_part.mean()
            if (mean_low > mean_high):
                std_smaller = high_part.std()
                mean_smaller = mean_high
                mean_higher = mean_low
                move_to = fir
            else:
                std_smaller = low_part.std()
                mean_smaller = mean_low
                mean_higher = mean_high
                move_to = sec

            if (mean_higher - (mean_smaller + 1 * std_smaller) > 0):
                lisT_save_tupple_years[bin_num] = (lisT_save_tupple_years[bin_num][0], move_to)
                lisT_save_tupple_years[bin_num + 1] = (move_to + 0.00001, lisT_save_tupple_years[bin_num + 1][1])

        elif (sec_empty == False and first_empty == True):
            low_part_2 = low_part_2[Y_axi]
            high_part_2 = high_part_2[Y_axi]
            mean_low_2 = low_part_2.mean()
            mean_high_2 = high_part_2.mean()
            if (mean_low_2 > mean_high_2):
                std_smaller = high_part_2.std()
                mean_smaller = mean_high_2
                mean_higher = mean_low_2
                move_to = fir
            else:
                std_smaller = low_part_2.std()
                mean_smaller = mean_low_2
                mean_higher = mean_high_2
                move_to = sec

            if (mean_higher - (mean_smaller + 1 * std_smaller) > 0):
                lisT_save_tupple_years[bin_num] = (lisT_save_tupple_years[bin_num][0], move_to)
                lisT_save_tupple_years[bin_num + 1] = (move_to + 0.000001, lisT_save_tupple_years[bin_num + 1][1])

        elif (sec_empty == True and first_empty == True):
            continue
        else:
            # case both true
            low_part = low_part[Y_axi]
            high_part = high_part[Y_axi]
            mean_low = low_part.mean()
            mean_high = high_part.mean()
            low_part_2 = low_part_2[Y_axi]
            high_part_2 = high_part_2[Y_axi]
            mean_low_2 = low_part_2.mean()
            mean_high_2 = high_part_2.mean()
            fir_sub = abs(mean_low - mean_high)
            fir_sub_2 = abs(mean_low_2 - mean_high_2)
            if (fir_sub > fir_sub_2):
                if (mean_low > mean_high):
                    std_smaller = high_part.std()
                    mean_smaller = mean_high
                    mean_higher = mean_low
                    move_to = fir
                else:
                    std_smaller = low_part.std()
                    mean_smaller = mean_low
                    mean_higher = mean_high
                    move_to = sec
                if (mean_higher - (mean_smaller + 1 * std_smaller) > 0):
                    lisT_save_tupple_years[bin_num] = (lisT_save_tupple_years[bin_num][0], move_to)
                    lisT_save_tupple_years[bin_num + 1] = (move_to + 0.00001, lisT_save_tupple_years[bin_num + 1][1])

            else:
                if (mean_low_2 > mean_high_2):
                    std_smaller = high_part_2.std()
                    mean_smaller = mean_high_2
                    mean_higher = mean_low_2
                    move_to = fir
                else:
                    std_smaller = low_part_2.std()
                    mean_smaller = mean_low_2
                    mean_higher = mean_high_2
                    move_to = sec

                if (mean_higher - (mean_smaller + 1 * std_smaller) > 0):
                    lisT_save_tupple_years[bin_num] = (lisT_save_tupple_years[bin_num][0], move_to)
                    lisT_save_tupple_years[bin_num + 1] = (move_to + 0.00001, lisT_save_tupple_years[bin_num + 1][1])

    bins = pd.IntervalIndex.from_tuples(cutNumbersForGraph(lisT_save_tupple_years))
    if should_print:
        result_no_outlay.groupby(pd.cut(result_no_outlay[X_axi], bins))[Y_axi].mean().plot(kind='line', ax=ax)

    if is_test_mode == True:
        testset = testset[[X_axi, Y_axi]]
        records = testset.to_records(index=False)
        test_dots = list(records)
        graph_dots = result_no_outlay.groupby(pd.cut(result_no_outlay[X_axi], bins))[Y_axi].mean()
        tupple_list_x_y_dot_graph = []
        # print(graph_dots)
        # print("boo1")
        # print(graph_dots.shape)
        # print("boo2")

        # for t in graph_dots:
        #   print("Naor 1")
        #   print(t)
        #   print("Naor 2")
        #   print(t.shape)

        # for i in range(0, len(lisT_save_tupple_years)):
        #   print(graph_dots[i])
        #   print("index: "+str(i))
        #   tupple_list_x_y_dot_graph.append((lisT_save_tupple_years[i][1], graph_dots[i]))

        i_for_graph_dots = 0
        for t in graph_dots:
          tupple_list_x_y_dot_graph.append((lisT_save_tupple_years[i_for_graph_dots][1], t))
          i_for_graph_dots = i_for_graph_dots+1


        distanceSum = 0
        for dot in test_dots:
            sortestDistance = 0
            for j in range(0, len(tupple_list_x_y_dot_graph) - 1):

                distance = np.abs(
                    np.linalg.norm(np.cross(np.subtract(tupple_list_x_y_dot_graph[j + 1], tupple_list_x_y_dot_graph[j])
                                            , np.subtract((dot[0], dot[1]),
                                                          tupple_list_x_y_dot_graph[j])))) / np.linalg.norm(
                    np.subtract(tupple_list_x_y_dot_graph[j + 1], tupple_list_x_y_dot_graph[j]))
                if j == 0:
                    sortestDistance = distance
                else:
                    if distance < sortestDistance:
                        sortestDistance = distance
            distanceSum += sortestDistance
        avgSortestDistances = distanceSum / len(test_dots)

        return (avgSortestDistances)





def bin_graph(n_clusters, X_axi, Y_axi, dtf, hight_y=4, width_x=10, test_mode=False):
    dtf = shuffle(dtf)
    len_dtf = len(dtf)
    validation_size = int(len_dtf * 0.1)
    test = dtf.iloc[:validation_size, :]
    train = dtf.iloc[validation_size:, :]
    i = 0
    saver_min_average_score_1 = 0
    save_best_theard_1 = 0
    save_best_num_std_1 = 1.5
    while i < 0.5:
        avgSortestDistances = basicBinsGraph(n_clusters, X_axi, Y_axi, train, is_test_mode=True, testset=test, theards=i,num_std=save_best_num_std_1,
                                        should_print=False)
        if i == 0:
            saver_min_average_score_1 = avgSortestDistances
            save_best_theard_1 = i
        elif avgSortestDistances < saver_min_average_score_1:
            saver_min_average_score_1 = avgSortestDistances
            save_best_theard_1 = i
        i = i + 0.1

    i = 0
    saver_min_average_score_2 = 0
    save_best_theard_2 = 0
    save_best_num_std_2 = 15
    while i < 0.5:
        avgSortestDistances = basicBinsGraph(n_clusters, X_axi, Y_axi, train, is_test_mode=True, testset=test,num_std=save_best_num_std_2,
                                             theards=i,
                                             should_print=False)
        if i == 0:
            saver_min_average_score_2 = avgSortestDistances
            save_best_theard_2 = i
        elif avgSortestDistances < saver_min_average_score_2:
            saver_min_average_score_2 = avgSortestDistances
            save_best_theard_2 = i
        i = i + 0.1

    if saver_min_average_score_2 < saver_min_average_score_1:
        basicBinsGraph(n_clusters, X_axi, Y_axi, dtf, hight=hight_y, width=width_x, is_test_mode=False, theards=save_best_theard_2,num_std=save_best_num_std_2,
          should_print=True)
        if test_mode:
            print("saver_min_average_score")
            print(saver_min_average_score_2)
    else:
        basicBinsGraph(n_clusters, X_axi, Y_axi, dtf, hight=hight_y, width=width_x, is_test_mode=False,
                       theards=save_best_theard_1, num_std=save_best_num_std_1,
                       should_print=True)
        if test_mode:
            print("saver_min_average_score")
            print(saver_min_average_score_1)







