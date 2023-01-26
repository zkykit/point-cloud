import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet


class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output


def sort_key_by_vale(key, value):
    assert key.shape == value.shape       #assert 断言操作，用于判断一个表达式，在表达式条件为false的时候触发异常
    assert len(key.shape) == 1            #numpy是多维数组
    sorted_idx = np.argsort(value)        #对value值进行排序
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]      #进行升序排序
    return key_sorted, value_sorted


def axis_round_robin(axis, dim):         #用于轴的轮换
    if axis == dim-1:
        return 0
    else:
        return axis + 1


def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):    #kd树的建立
    """
    :param root:
    :param db: NxD
    :param db_sorted_idx_inv: NxD
    :param point_idx: M
    :param axis: scalar
    :param leaf_size: scalar
    :return:
    """
    if root is None:
        root = Node(axis, None, None, None, point_indices)           #实例化Node (key,left,right)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:                              #判断是否需要进行分割
        # --- get the split position ---
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  #对点进行排列，dp存储信息
   
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1     #分一半
 
        middle_left_point_idx = point_indices_sorted[middle_left_idx]          #左边界点
       
        middle_left_point_value = db[middle_left_point_idx, axis] #输出是0~1
       
        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]           #右边界点

        root.value = (middle_left_point_value + middle_right_point_value) * 0.5    #取中值为节点值 (0~1)
        
        # === get the split position ===
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0:middle_right_idx],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)                                  #对对应的轴值进行排序
        root.right = kdtree_recursive_build(root.right,
                                           db,
                                           point_indices_sorted[middle_right_idx:],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)                                  #对对应的轴值进行排序
    return root


def traverse_kdtree(root: Node, depth, max_depth):      #计算kdtree的深度
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():                                 #打印最后的叶子节点
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)    #累加计算深度
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1


def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):   #KNNResultSet 继承二叉树的结果集
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:          #如果 q[axis] inside the partition   如果查询点在根节点的左边，一定要查找左边
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():   #如果目标点离轴虚线的距离小于worst_dist 继续搜寻节点的右边
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    return False


def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set, query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set, query)

    return False



def main():
    # configuration
    db_size = 64
    dim = 3                #三维
    leaf_size = 4
    k = 1                  #一个点

    db_np = np.random.rand(db_size, dim)

    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # knn_search(root, db_np, result_set, query)
    #
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    # print("Radius search:")
    # query = np.asarray([0, 0, 0])
    # result_set = RadiusNNResultSet(radius = 0.5)
    # radius_search(root, db_np, result_set, query)
    # print(result_set)


if __name__ == '__main__':
    main()
