import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
# Tọa độ của các điểm
data_x = [ (1, 4), (2, 3), (4, 2), (5, 5)]
weight_data_x =[-3, 2, 1, 0]
# data_x_ot = 3 # đường xương sống
# Tạo đồ thị có hướng
G = nx.DiGraph()
# Thêm các đỉnh vào đồ thị và gán trọng số
def add_node (list_of_points):
   for i in range(len(list_of_points)):
      G.add_node(list_of_points[i][0], pos = list_of_points[i][1], weight= list_of_points[i][2] )
# Hàm tính khoảng cách Euclid
def Mahatan_distance(point1, point2):
    return np.abs((point1[0] - point2[0])) + np.abs(point1[1] - point2[1])

#Ham noi hinh chieu voi x
def add_canh_proj_x (danh_sach_proj,danh_sach_bien):
   for i in range(len(danh_sach_proj)):
      G.add_edge (danh_sach_proj[i][0], danh_sach_bien[i][0], weight = Mahatan_distance(danh_sach_proj[i][1], danh_sach_bien[i][1]))

# Hàm DFS để xác định cây con
def dfs_subtree(graph, start_node):
    visited = set()
    stack = [start_node]
    subtree_nodes = set()

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            subtree_nodes.add(node)
            stack.extend([n for n in graph.successors(node) if n not in visited])

    return subtree_nodes

def tinh_OT(x, weight_x, x_ot):
    
    #tạo hai danh sách biến và hình chiếu tương ứng data_ponint, projecton
    # danh sach bien
    data_point =[]
    for i in range(len (x) ):
        data_point.append( (f'x{i+1}',x[i], weight_x[i])) 


    #hình chiếu
    projections = []

    # Tính toán các điểm hình chiếu và đặt tên
    for i, point in enumerate(x , start=1):
        # Giữ nguyên y và thay đổi x thành 2.5
        projected_point = (x_ot, point[1])
        projections.append((f"p{i}", projected_point, 0))  # Thêm tuple vào danh sách

    # Sắp xếp và lấy tên biến
    sorted_proj = sorted(projections, key=lambda item: item[1][1], reverse=True)
    sorted_names = [item[0] for item in sorted_proj]
    p = sorted_proj



    add_node (data_point)
    add_node (projections)
    for i in range(len(p)-1):
        G.add_edge(p[i][0], p[i+1][0], weight=Mahatan_distance(p[i][1], p[i+1][1]))  # Kết nối p1 với p2, p2 với p3



    # Thêm cạnh nối giữa các điểm hình chiếu (tạo thành cây)
    add_canh_proj_x(projections, data_point)
    pos = nx.get_node_attributes(G, 'pos') # Lấy vị trí của các đỉnh
    edge_labels = nx.get_edge_attributes(G, 'weight') # Lấy trọng số của các cạnh
    node_weights = nx.get_node_attributes(G, 'weight') # Lấy trọng số của các đỉnh

    # Vẽ đồ thị
    #nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray', arrows=True)

    # Vẽ trọng số trên các cạnh
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')

    # Vẽ trọng số trên các đỉnh
    for node, (x, y) in pos.items():
        plt.text(x, y + 0.15, f'w={node_weights[node]}', fontsize=9, ha='center', color='darkgreen')

    # Vẽ đường thẳng x = 2.5
    plt.axvline(x= x_ot, color='purple', linestyle='--', label= f'x = {x_ot}')

    # Hiển thị chú thích
    plt.legend()

    # Đặt tiêu đề cho đồ thị
    plt.title("Đồ thị cây có hướng với trọng số các đỉnh và cạnh")


    # Xác định gốc của cây lớn ban đầu (điểm hình chiếu có tung độ lớn nhất)
    # Lấy danh sách các nút hình chiếu
    projection_nodes = [node for node in G.nodes if node.startswith('p')]

    # Xác định gốc của cây lớn ban đầu
    root = max(projection_nodes, key=lambda k: G.nodes[k]['pos'][1])

    # Xác định cây lớn ban đầu có gốc là root (p1)
    tree_nodes = dfs_subtree(G, root)

    # Xác định cây con có gốc là p3
    subtree_root = "p2"
    subtree_nodes = dfs_subtree(G, subtree_root)

    # Tính tổng trọng số của cây con
    subtree_weight = sum(G.nodes[node]['weight'] for node in subtree_nodes)
    

    # In ra độ dài cạnh (p1, x1)
    edge_length_p1_x1 = G["p1"]["x1"]["weight"]

    # In ra trọng số của các đỉnh x1, x2, x3
    dinh = data_point + projections
    ds_dinh = []
    for key, value, weight in dinh:
        ds_dinh.append(key)
    subtree_weight_matrix = []
    for subtree_root in ds_dinh:
        subtree_nodes = dfs_subtree(G, subtree_root)
        subtree_weight = sum(G.nodes[node]['weight'] for node in subtree_nodes)
        subtree_weight_matrix.append(subtree_weight)
    subtree_weight_matrix = np.array(subtree_weight_matrix)
    # Tạo từ điển để lưu trọng số của các cạnh đi vào từng đỉnh
    incoming_weights = {}

    # Tính toán trọng số của các cạnh đi vào từng đỉnh
    for node in ds_dinh:
        incoming_edges = G.in_edges(node, data=True)  # Lấy các cạnh đi vào đỉnh
        if incoming_edges:  # Nếu có cạnh đi vào
            total_weight = sum(data['weight'] for _, _, data in incoming_edges)
        else:  # Nếu không có cạnh đi vào
            total_weight = 0
        incoming_weights[node] = total_weight


        # Tạo ma trận với một dòng chứa các trọng số
    weights_array = np.array([incoming_weights[node] for node in ds_dinh])

    subtree_weight_matrix_abs = np.abs(subtree_weight_matrix) # lay gia tri tuyet doi trong so cay
    OT= np.dot(weights_array, subtree_weight_matrix_abs.T)
    # print(OT)
    return OT
    #plt.show()



x_xuong_song= []
S = []
for u, v in data_x:
    x_xuong_song.append(u)
# print(x_xuong_song)

for x_ot in x_xuong_song:
    A = tinh_OT(data_x, weight_data_x,x_ot)
    S.append(A)
# print(S)    
value_OT= min(S)

# Tìm tất cả vị trí của phần tử nhỏ nhất
min_indices = [index for index, value in enumerate(S) if value == value_OT]
for i in min_indices:
    print(' gia tri x de dat OT la: ', x_xuong_song[i])
# Hiển thị kết quả
print("OT là",value_OT )