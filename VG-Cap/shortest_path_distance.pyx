# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy

#距离矩阵D_ij取值范围:[0,2m+1].其中 0：对角线元素; 1-6：正向距离; 7-12：反向距离; 13:正向和反向都不可到达，不连接。

def floyd_warshall(adjacency_matrix, direct_matrix):  #parameter：不区分方向的对称的邻接矩阵  区分方向的邻接矩阵
#佛洛依德算法
    (batchsize, nrows, ncols) = adjacency_matrix.shape
    shortest_dis = numpy.zeros([batchsize, nrows, ncols])  #置0
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef numpy.ndarray[long, ndim=2, mode='c'] M
    cdef numpy.ndarray[long, ndim=2, mode='c'] path
    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj     #cost_ikkj表示代价
    cdef long* M_ptr  #指针操作
    cdef long* M_i_ptr
    cdef long* M_k_ptr
    for BatchNum in range(batchsize):
        oneBatch = adjacency_matrix[BatchNum].astype(numpy.int64)
        adj_mat_copy = oneBatch.astype(long, order='C', casting='safe', copy=True)
        assert adj_mat_copy.flags['C_CONTIGUOUS']
        M = adj_mat_copy  #邻接矩阵
        path = numpy.zeros([n, n], dtype=numpy.int64)#path[i,j]表示从点i到点j经过的第一个点的编号，比如从点i到点j的最短路径上，依次经过点i、点a、点b、点j，那么path[i,j]存储的值为a
        M_ptr = &M[0,0]
        # set unreachable nodes distance to 101, 取的一个最大值，object的特征个数为0-100.
        for i in range(n):
            for j in range(n):
                path[i][j] = j    #假设点i与点j直接相连，所以从i到j经过的第一个点就是j
                if i == j:
                    M[i][j] = 0  #对角线值不连接
                elif M[i][j] == 0: #除对角线之外
                    M[i][j] = 101   # 不相连
                    path[i][j] = -1  #表示此时i与j不直接相连，在只考虑距离为1时，没有可以经过的点，所以设为-1，这个值后续可以改变

        # floyed algo 算法主体结构
        for k in range(n):
            M_k_ptr = M_ptr + n*k
            for i in range(n):
                M_i_ptr = M_ptr + n*i
                M_ik = M_i_ptr[k]
                for j in range(n):
                    cost_ikkj = M_ik + M_k_ptr[j]
                    M_ij = M_i_ptr[j]
                    if M_ij > cost_ikkj:
                        M_i_ptr[j] = cost_ikkj   #i到j的最短距离，直接由指针修改地址对应的值
                        path[i][j] = path[i][k]  #表示i与j虽然不直接连接，但是能间接相连，i到j经过的第一个点的序号，与i到k经过的第一个点的序号相同

        for i in range(n):
            for j in range(n):
                if M[i][j] >= 6 and M[i][j] < 101:
                    M[i][j] = 6
                elif M[i][j] >= 101:
                    M[i][j] = 13
                if i == j:
                    M[i][j] = 0
        #考虑方向性
        for i in range(n):
            for j in range(n):
                if path[i][j] != -1: #不等于-1表示，i至少是能够到j的，存储的值表示i到j经过的第一个点的标号。这里只能区分点i、点a、点b、点j；i->a的正反向。
                    if direct_matrix[BatchNum, i, path[i][j]] != 1: #path[i][j]表示i到j时经过的第一个点的标号，设为a，那么如果i到点a在有向图中直接相连（等于1），这就意味着是正向的距离，距离加上7以示区分，否则是反向
                        if M[i][j] != 0 and M[i][j] != 13:
                            M[i][j] = M[i][j] + 6 #反向加6，正向不加  所有取值0~13： 0：对角线元素 1-6：正向距离 7-12：反向距离 13:正向和反向都不可到达，不连接

        shortest_dis[BatchNum] = M
    shortest_dis = numpy.array(shortest_dis).astype(numpy.int64)

    return shortest_dis
