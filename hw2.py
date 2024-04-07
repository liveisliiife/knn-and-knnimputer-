import numpy as np
import pandas as pd

def calculate_distances(distance_method,existing_data, test_data):
    distances = []
    for i in range(len(existing_data)):
        x1 = existing_data.iloc[i, 1:]
        x2 = test_data
        distance = 0
        if distance_method == 'euclidean':
            distance = np.sqrt(np.sum((x1 - x2) ** 2))
        elif distance_method == 'manhattan':
            distance = np.sum(np.abs(x1 - x2))
        elif distance_method == 'chebyshev':
            distance = np.max(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported distance method.")
        
        distances.append((distance, existing_data.iloc[i, 0]))  # (distance, label)

    return distances


def weighted_voting_function(distances, k):
    labels = []
    weights = []

    for dist, label in distances[:k]:
        if dist == 0:
            weight = 1 / 0.000000001
        else:
            weight = 1 / dist**2     
        if label in labels:
            index = labels.index(label)
            weights[index] += weight
        else:
            labels.append(label)
            weights.append(weight)

    if len(weights) == 0:
      raise ValueError("weights list is empty")   
     
    max_weight_index = np.argmax(weights)
    prediction = labels[max_weight_index]

    return prediction





def knn(existing_data, test_data, k, distance_method, re_training, distance_threshold=None, weighted_voting=False):
    predictions = []

    distance_method = distance_method.lower()  # Euclidean(?) ---> euclidean  , euclidean ---> euclidean

    if existing_data.shape[1] == test_data.shape[1]:   # test_data should be unlabeled
      test_data = test_data.iloc[:,1:]


    for i in range(len(test_data)):

        distances = calculate_distances(distance_method,existing_data, test_data.iloc[i,:])   
        distances.sort()  
        
        if distance_threshold is not None:
          if distance_threshold <=0:
            raise ValueError("distance_threshold should be positive float number")
          temp_distance = []
          for dist in distances:
            if dist[0] <= distance_threshold:
              temp_distance.append(dist)
          if len(temp_distance)==0:
            raise ValueError("increase the distance_threshold")

          distances = temp_distance


        if weighted_voting == True:
            prediction = weighted_voting_function(distances,k)
        else:
          labels = []
          for dist,label in distances[:k]:
            labels.append(label)
          prediction = max(set(labels), key=labels.count)
        predictions.append(prediction)


        if re_training == True:
            new_row = [prediction] + test_data.iloc[i,:].values.tolist()
            new_row_df = pd.DataFrame([new_row], columns=existing_data.columns)
            existing_data = pd.concat([existing_data, new_row_df], ignore_index=True)
            #print(existing_data.shape)
            
    return pd.Series(predictions)



def fill_missing_features(existing_data, test_data,k, distance_method, distance_threshold, weighted_voting):
  
  distance_method = distance_method.lower()  # Euclidean(?) ---> euclidean  , euclidean ---> euclidean

  test_missing = test_data.copy()

  for i in range(len(test_data)): 

    test_label = test_data.iloc[i, 0]                                         # label for test_data  ---> 0,1,2...

    existing_data_same = existing_data[existing_data.iloc[:,0] == test_label]    # those with the same label as the i.th row of test_data

    test_missing_unlabeled = test_data.iloc[i,1:]                                 # remove the label and take the i.th row

    nan_column_names = test_missing.iloc[i].index[test_missing.iloc[i].isna()].tolist()

    nan_column_name = nan_column_names[0]                      # nan column name  f3

    column_index = existing_data.columns.get_loc(nan_column_name)   # nan column index  3


    distances = []
    for j in range(len(existing_data_same)):
      x1 = existing_data_same.iloc[j, 1:]
      x2 = test_missing_unlabeled
      distance = 0
      if distance_method == 'euclidean':
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
      elif distance_method == 'manhattan':
        distance = np.sum(np.abs(x1 - x2))
      elif distance_method == 'chebyshev':
          distance = np.max(np.abs(x1 - x2))
      else:
          raise ValueError("Unsupported distance method.")
        
      distances.append((distance, j))  # (distance, index_number)


    distances.sort()                   # sort


    if distance_threshold is not None:
      if distance_threshold <=0:
          raise ValueError("distance_threshold should be positive float number")
      temp_distance = []
      for dist in distances:
        if dist[0] <= distance_threshold:
            temp_distance.append(dist)
        if len(temp_distance)==0:
          raise ValueError("increase the distance_threshold")
      distances = temp_distance
  
    prediction_nan = 0

    if weighted_voting == False:
      sum = 0
      for dist,index in distances[:k]:
        value = existing_data.iloc[index, column_index]
        sum = sum  + value
      prediction_nan = sum / k

    if weighted_voting == True:
      total_weight = 0
      weighted_sum = 0
      for dist, index in distances[:k]:
        weight = 1 / (dist ** 2)  
        value = existing_data.iloc[index, column_index]
        weighted_sum += weight * value
        total_weight += weight
      prediction_nan = weighted_sum / total_weight


    test_missing.iloc[i,column_index] = prediction_nan

  return test_missing



"""
test = pd.read_csv("C:/Users/pc/Downloads/test_normalized_v2.csv")
train = pd.read_csv("C:/Users/pc/Downloads/train_normalized_v2.csv")
test_missing = pd.read_csv("C:/Users/pc/Downloads/test_with_missing_normalized_v2.csv")
# Task 1
predictions_task1 = knn(train, test, k=2, distance_method='Euclidean', re_training=True, distance_threshold=15.2, weighted_voting=True)
print("Predictions for Task 1:")
print(predictions_task1)
print((test.label != predictions_task1).sum())


# Task 2
predictions_task2 = fill_missing_features(train, test_missing, k=2, distance_method='Euclidean', distance_threshold=15.2, weighted_voting=True)
print("\nPredictions for Task 2:")
print(predictions_task2)

# Task 2 ---> Task 1
print("final result")
predictions_task1 = knn(train, predictions_task2, k=2, distance_method='Euclidean', re_training=True, distance_threshold=15.2, weighted_voting=True)
print("Predictions for Task 1:")
print(predictions_task1)
print((test.label != predictions_task1).sum())
"""





