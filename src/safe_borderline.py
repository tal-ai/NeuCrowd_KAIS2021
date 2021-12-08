import numpy as np
import heapq
 
'''
Finding safe instances to be trained, 
based on assurance scores of instances in the neighborhood.

'''


def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def safe_borderline(X, X_label, confidence):

    # X: total_data_size * dim
    #
    ED = EuclideanDistances(X, X)  # (total_data_size, total_data_size)

    data_size = ED.shape[0]

    # how many data point are included.
    n = np.int(np.sqrt(data_size)) + 1

    safety_list = []
    unsafe_list = []
    is_safe = []

    for i in range(data_size):
        least_dist_index = heapq.nsmallest(n, range(data_size), ED[i, :].take)
        nearest_confidence = confidence[least_dist_index]
        # print('\n')

        x = X_label[i]
        # print('assurance of instances in the same class')
        # print(nearest_confidence[X_label[least_dist_index] == x])
        # print('assurance of instances in different class')
        # print(nearest_confidence[X_label[least_dist_index] != x])

        if np.sum(nearest_confidence[X_label[least_dist_index] == x]) >= \
        np.sum(nearest_confidence[X_label[least_dist_index] != x]):
            # print('x is safe')
            safety_list.append(i)
            is_safe.append(1)
        else:
            # print('x is dangerous')
            unsafe_list.append(i)
            is_safe.append(0)
    print('safe count: ', sum(is_safe))
    return safety_list, unsafe_list, is_safe


if __name__ == '__main__':
    a = np.array([[1, 1, 1, 1, 1],
                  [-1, -1, -1, -1, -1],
                  [2, 2, 2, 2, 2],
                  [3, 3, 3.1, 3, 3],
                  [4, 4, 4, 4, 4]]
                 )
    b = np.array([1, -1, 1, -1, -1])
    c = np.array([0.3, 0.8, 0.6, 0.5, 0.7])
    print(safe_borderline(a, b, c))
