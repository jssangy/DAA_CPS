import copy
import heapq
import numpy as np # use for matrix calculation
from sklearn import neighbors  # use for dijkstra

import Funct

class controller():
    
    def __init__(self, agv_num, map):
        self.agv_pos = {} # save the position of agv positions
        self.agv_next_pos = {} # save the next position of agv positions
        self.agv_prev_pos = {} # save the previous position of agv positions
        self.agv_next_rout = {} # save the next rout position of agv
        self.control_buffer = {} # save the control output of agvs
        self.agv_state = {} # 0(start - pick up) 1(pick up - drop) 2(drop - rest) 3(rest - start)
        self.agv_nums = [] # agv numbers (A, B, C, ... O)
        self.agv_mode = {} # 0 (normal) 1 (Danger)
        self.agv_goal = {} # goal position of all agvs
        self.agv_info = {} # for GUI infomation
        self.agv_rout = {} # for routing of AGV
        self.agv_prev_rout = {} # previous node
        self.optimal_time = {} # Optimal time to reach goal
        self.actual_time = {} # Actual time to reach goal
        self.zone = set()
        self.matrix_idx = []
        self.path_matrix = []
        self.zone_matrix = {}
        
        self.running_opt = 0
        
        # Whole Products
        self.whole_product = 0
            
        # Initialization
        for i in range (agv_num):
            self.agv_nums.append(chr(i + 65))
            self.agv_pos[chr(i + 65)] = (0, 0)
            self.agv_prev_pos[chr(i + 65)] = (0, 0)
            self.agv_next_rout[chr(i + 65)] = (0, 0)
            self.agv_state[chr(i + 65)] = 0 # Initial state is 0
            self.agv_mode[chr(i + 65)] = 0 # Initial mode is normal
            self.agv_goal[chr(i + 65)] = [(0, 0), (0, 0), (0, 0), (0, 0)]
            self.agv_info[chr(i + 65)] = [0, 0]
            self.control_buffer[chr(i + 65)] = (0, 0)
            self.agv_rout[chr(i + 65)] = []
            self.agv_prev_rout[chr(i + 65)] = (0, 0)
            self.optimal_time[chr(i + 65)] = 0
            self.actual_time[chr(i + 65)] = 0
        
        # Map of warehouse digital twin
        self.map = map
        
        # Make graph for routing
        self.graphing()
        
        # Make zone for routing
        self.make_zone()
        
        #print("Node: ", len(self.graph), " Zone: ", len(self.zone))
        
        # Make path matrix for routing
        self.make_path_matrix()
        
        # Time
        self.time = 0
    
    # set start position
    def set_start(self, num, pos):
        self.agv_goal[num][3] = pos
    
    # set pick-up position
    def set_pick(self, num, pos):
        self.agv_goal[num][0] = pos
        
    # set drop position
    def set_drop(self, num, pos):
        self.agv_goal[num][1] = pos
        
    # set rest position
    def set_rest(self, num, pos):
        self.agv_goal[num][2] = pos
    
    # Change the AGV's state
    def change_state(self, num, state):
        if state != 3:
            self.agv_state[num] = state + 1   
        else:
            self.agv_state[num] = 0
            self.agv_info[num][0] += 1
            self.whole_product += 1
        
        return self.agv_state[num]
        
    # Update data from sensing of agv
    def get_sensing(self, num, data):
        if data != None:
            self.agv_pos[num] = data[0]
            self.agv_mode[num] = data[1]
            self.agv_info[num][1] = data[1]
            
        if self.time == 0:
            self.agv_rout[num] = self.dijkstra_shortest(self.graph, self.agv_pos[num], self.agv_goal[num][self.agv_state[num]])
            self.agv_prev_rout[num] = self.agv_pos[num]
            self.agv_next_rout[num] = self.agv_rout[num][0]
    
    def make_control(self):
        self.time += 1
        self.dijkstra_rout()
        return (self.control_buffer, self.agv_mode)
    
    # Get AGV's State for Reinforcement Learning (Optimized with NumPy)
    def get_state(self):
        num_agvs = len(self.agv_nums)
        state = np.full((num_agvs, 15), -1, dtype=int)  # 기본값 -1로 채워진 배열 생성

        for i, agv_id in enumerate(self.agv_nums):
            pos = self.agv_pos[agv_id]
            goal_pos = self.agv_goal[agv_id][self.agv_state[agv_id]]
            remaining_nodes = self.agv_rout[agv_id][:5]

            # 값 대입
            state[i, 0:2] = pos
            state[i, 2] = self.agv_mode[agv_id]
            
            # 남은 노드 값 입력
            for j, node in enumerate(remaining_nodes):
                state[i, 3 + j*2: 5 + j*2] = node

            # 목표 위치 대입
            state[i, 13:15] = goal_pos

        return state  # shape: (N, 15)
    
    # Perform extra Action when event occur (Reinforcement Learning)
    def perform_action(self, agv_id, action):
        if action == 0: # Follow
            pass
        elif action == 1: # Replan
            self.replan_rout(agv_id)
        elif action == 2: # Wait
            self.wait(agv_id)
    
    # Action: Replan AGV rout
    def replan_rout(self, agv_id, edge_penalty=100, num_penalty_edges=1):
        # 현재 위치 및 목적지 확인
        current_pos = self.agv_pos[agv_id]
        current_goal = self.agv_goal[agv_id][self.agv_state[agv_id]]

        # 임시 그래프 복사본 생성
        temp_graph = copy.deepcopy(self.graph)

        # 기존에 계획된 경로 가져오기
        remaining_rout = self.agv_rout[agv_id]

        # AGV가 엣지 위에 있는지 체크 (노드 목록에 없으면 엣지 위로 판단)
        if current_pos not in temp_graph:
            start_node = self.agv_prev_rout[agv_id]
        else:
            start_node = current_pos

        # 현재 위치에서 시작하여 가까운 N개의 edge 가중치를 높임
        affected_edges = []

        for next_node in remaining_rout[:num_penalty_edges]:
            affected_edges.append((start_node, next_node))
            start_node = next_node

        # affected_edges의 가중치를 높임
        for node1, node2 in affected_edges:
            if node2 in temp_graph[node1]:
                temp_graph[node1][node2] = edge_penalty
            if node1 in temp_graph[node2]:
                temp_graph[node2][node1] = edge_penalty

        # 경로 재계산
        new_rout = self.dijkstra_shortest(temp_graph, start_node, current_goal)

        # 재계산 결과 확인
        if new_rout == -1 or len(new_rout) == 0:
            print(f"경로 재계산 실패: AGV {agv_id}, 현재 위치 {current_pos}, 시작 위치 {start_node}, 목표 위치 {current_goal}")
            return False  # 실패
        else:
            self.agv_rout[agv_id] = new_rout
            return True  # 성공

    # Action: Wait AGV    
    def wait(self, agv_id):
        self.control_buffer[agv_id] = (0, 0)
        self.agv_mode[agv_id] = (0, 0)
    
    # 수정 절대 금지! (수정해버림 ㅋ)
    def dijkstra_rout(self):
        # Get the Dijkstra rout of AGVs
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]
            prev_pos = self.agv_prev_pos[num]

            if (prev_pos in self.agv_goal[num] and pos != prev_pos):
                self.actual_time[num] = 1 # Actual time initialization
                
                # Compute optimal time to reach goal
                path = self.dijkstra_shortest(self.graph, prev_pos, goal)
                self.optimal_time[num] = sum(Funct.get_distance(path[i], path[i+1]) for i in range(len(path)-1))
            else:
                self.actual_time[num] += 1

            # Change the state of AGVs
            if (pos == goal):
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0
                self.agv_rout[num] = self.dijkstra_shortest(self.graph, pos, goal)
                self.agv_prev_rout[num] = pos
            
            # If AGV need next rout! (rout node)
            if (((self.map[pos[1]][pos[0]] == 6) or (pos in self.agv_goal[num])) and (self.agv_mode[num] == 0)):
                self.agv_prev_rout[num] = pos
                next_rout = self.agv_rout[num].pop(0)
                
                # Save next rout position
                self.agv_next_rout[num] = next_rout
                
                # Determine new control signal
                if next_rout[0] > pos[0]:
                    self.control_buffer[num] = (1, 0)
                elif next_rout[0] < pos[0]:
                    self.control_buffer[num] = (-1, 0)
                elif next_rout[1] > pos[1]:
                    self.control_buffer[num] = (0, 1)
                elif next_rout[1] < pos[1]:
                    self.control_buffer[num] = (0, -1)
                else:
                    self.control_buffer[num] = (0, 0)
                self.agv_next_pos[num] = (pos[0] + self.control_buffer[num][0], pos[1] + self.control_buffer[num][1]) 
        
            # Just keep going!
            else:
                self.agv_next_pos[num] = (pos[0] + self.control_buffer[num][0], pos[1] + self.control_buffer[num][1]) 
                
        # Collision prevention => Dead Lock
        for num1 in self.agv_nums:
            num1_pos = self.agv_pos[num1]
            num1_next_pos = self.agv_next_pos[num1]

            # Deadlock 상태 초기화
            self.agv_mode[num1] = 0

            for num2 in self.agv_nums:
                if num1 != num2:
                    num2_pos = self.agv_pos[num2]
                    num2_next_pos = self.agv_next_pos[num2]
                    num2_control_buffer = self.control_buffer[num2]
                    if (num1_next_pos == num2_next_pos):
                        self.agv_mode[num1] = 1
                    elif (num1_next_pos == num2_pos and num2_next_pos == num1_pos):
                        self.agv_mode[num1] = 1
                    elif (num1_next_pos == num2_pos and num2_control_buffer == (0, 0)):
                        self.agv_mode[num1] = 1
          
            if self.map[num1_pos[1]][num1_pos[0]] == 1:
                self.agv_mode[num1] = 2
                self.control_buffer[num1] = (0, 0)    
        
    # ======================== Routing Functions ============================================
    def graphing(self):
        self.graph = {}
        for x in range (100):
            for y in range(100):
                # normal node 
                if self.map[y][x] == 6:
                    neighbors = self.find_neighbors(x,y)
                    nodes = {}
                    for neighbor in neighbors:
                        nodes[neighbor] = Funct.get_distance((x,y), neighbor)
                    self.graph[(x,y)] = nodes
                    
                if type(self.map[y][x]) == str:
                    neighbors = self.find_neighbors(x,y, False)
                    nodes = {}
                    for neighbor in neighbors:
                        nodes[neighbor] = Funct.get_distance((x,y), neighbor)
                    self.graph[(x,y)] = nodes   
    
    def find_neighbors(self, x, y, rout = True):
        line_list = []
        distance = 0
        poss_x = x
        poss_y = y
        # up
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_x += 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
                
        distance = 0
        poss_x = x
        poss_y = y
        
        # down
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_x -= 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
        
        distance = 0
        poss_x = x
        poss_y = y
        
        # right
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_y += 1
            distance += 1
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
            if (self.map[poss_y][poss_x] == 1):
                break
            
        distance = 0
        poss_x = x
        poss_y = y
        
        # left
        while distance < 15 and 1 <= poss_y < 99 and 1 <= poss_x < 99:
            poss_y -= 1
            distance += 1
            if (self.map[poss_y][poss_x] == 1):
                break
            if (self.map[poss_y][poss_x] == 6):
                line_list.append((poss_x, poss_y))
                break
            if (type(self.map[poss_y][poss_x]) == str) and rout:
                line_list.append((poss_x, poss_y))
                break
            
        return line_list

    def dijkstra_shortest(self, graph, start, end):
        distances = {node: float('inf') for node in graph}  # start로 부터의 거리 값을 저장하기 위함
        distances[start] = 0  # 시작 값은 0이어야 함
        queue = []
        heapq.heappush(queue, [distances[start], start])  # 시작 노드부터 탐색 시작 하기 위함.
        
        parents = {start: None}
        distance = {start: 0}

        while queue:  # queue에 남아 있는 노드가 없으면 끝
            current_distance, current_destination = heapq.heappop(queue)  # 탐색 할 노드, 거리를 가져옴.

            if current_destination == end:
                return self.traceback_path(end, parents)

            if distances[current_destination] < current_distance:  # 기존에 있는 거리보다 길다면, 볼 필요도 없음
                continue
            
            for new_destination, new_distance in graph[current_destination].items():
                distance = current_distance + new_distance  # 해당 노드를 거쳐 갈 때 거리
                if distance < distances[new_destination]:  # 알고 있는 거리 보다 작으면 갱신
                    distances[new_destination] = distance
                    heapq.heappush(queue, [distance, new_destination])  # 다음 인접 거리를 계산 하기 위해 큐에 삽입
                    parents[new_destination] = current_destination
            
        return -1
    
    def traceback_path(self, target, parents):
        path = []
        while target:
            path.append(target)
            target = parents[target]
        return list(reversed(path))[1:]
    
    def make_zone(self):
        self.zone = set()
        for node, neighbors in self.graph.items():
            for neighbor in neighbors.keys():
                if ((node, neighbor) not in self.zone) and ((neighbor, node) not in self.zone):
                    self.zone.add((node, neighbor))
                    
    def make_path_matrix(self):
        dimension = len(self.zone) + len(self.agv_nums)
        entity = [0 for j in range(dimension)]
        for i in range(dimension):
            self.path_matrix.append(entity)
        self.path_matrix = np.array(self.path_matrix)
        for agv in self.agv_nums:
            self.matrix_idx.append(agv)
        for zone in self.zone:
            self.matrix_idx.append(zone)
            self.zone_matrix[zone] = 0
            
    def find_idx(self, node):
        if node in self.matrix_idx:
            idx = self.matrix_idx.index(node)
        else:
            node_reverse = (node[1], node[0])
            idx = self.matrix_idx.index(node_reverse)
        return idx
    
    def insert_edge(self, node1, node2):
        i = self.find_idx(node1)
        matrix_i1 = []
        matrix_i2 = []
        for i_t in range(len(self.path_matrix)):
            matrix_i1.append(self.path_matrix[i_t][i])
            if i_t == i:
                matrix_i2.append(1)
            else:
                matrix_i2.append(0)
        
        j = self.find_idx(node2)
        matrix_j1 = []
        matrix_j2 = []
        for j_t in range(len(self.path_matrix)):
            matrix_j1.append(self.path_matrix[j][j_t])
            if j_t == j:
                matrix_j2.append(1)
            else:
                matrix_j2.append(0)
        
        matrix_i1 = np.array(matrix_i1)
        matrix_i2 = np.array(matrix_i2)
        matrix_j1 = np.array(matrix_j1)
        matrix_j2 = np.array(matrix_j2)
        
        # matrix_i = np.append(matrix_i1, matrix_i2)
        # matrix_j = np.append(matrix_j1, matrix_j2)
        matrix_i = matrix_i1 + matrix_i2
        matrix_j = matrix_j1 + matrix_j2
        
        matrix_out = np.outer(matrix_i, matrix_j)
        
        self.path_matrix = self.path_matrix + matrix_out
        
    def delete_edge(self, node1, node2):
        i = self.find_idx(node1)
        matrix_i1 = []
        matrix_i2 = []
        for i_t in range(len(self.path_matrix)):
            matrix_i1.append(self.path_matrix[i_t][i])
            if i_t == i:
                matrix_i2.append(1)
            else:
                matrix_i2.append(0)
        
        j = self.find_idx(node2)
        matrix_j1 = []
        matrix_j2 = []
        for j_t in range(len(self.path_matrix)):
            matrix_j1.append(self.path_matrix[j][j_t])
            if j_t == j:
                matrix_j2.append(1)
            else:
                matrix_j2.append(0)
        
        matrix_i1 = np.array(matrix_i1)
        matrix_i2 = np.array(matrix_i2)
        matrix_j1 = np.array(matrix_j1)
        matrix_j2 = np.array(matrix_j2)
        
        # matrix_i = np.append(matrix_i1, matrix_i2)
        # matrix_j = np.append(matrix_j1, matrix_j2)
        matrix_i = matrix_i1 + matrix_i2
        matrix_j = matrix_j1 + matrix_j2
        
        matrix_out = np.outer(matrix_i, matrix_j)
        
        self.path_matrix = self.path_matrix - matrix_out
        
    
    def in_zone(self, zone, num):
        self.zone_matrix[zone] = num
    
    def out_zone(self, zone):
        self.zone_matrix[zone] = 0
        
    def idle_zone(self, zone, num):
        if zone in self.zone_matrix:
            pass
        else:
            zone = (zone[1], zone[0])
            
        if self.zone_matrix[zone] == 0 or self.zone_matrix[zone] == num:
            return True
        else:
            return False
        
    
    def zone_control(self, num, zone):
        i = self.find_idx(num)
        j = self.find_idx(zone)
        # No Cycle
        # if np.diaonal(self.path_matrix[i][j]) > 0:
        if np.diagonal(self.path_matrix) > 0:
            return True
        else:
            return False
        
    def alter_path(self, pos, origin_rout):
        for neighbor in self.graph[pos]:
            if neighbor != origin_rout:
                return neighbor