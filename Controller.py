import heapq
import numpy as np # use for matrix calculation
from sklearn import neighbors  # use for dijkstra

import Funct

class controller():
    
    def __init__(self, agv_num, map):
        
        self.agv_pos = {} # save the position of agv positions
        self.agv_next_pos = {} # save the next position of agv positions
        self.agv_next_rout = {} # save the next rout position of agv
        self.control_buffer = {} # save the control output of agvs
        self.agv_state = {} # 0(start - pick up) 1(pick up - drop) 2(drop - rest) 3(rest - start)
        self.agv_nums = [] # agv numbers (A, B, C, ... O)
        self.agv_mode = {} # 0 (normal) 1 (Danger)
        self.agv_goal = {} # goal position of all agvs
        self.agv_info = {} # for GUI infomation
        self.agv_rout = {} # for routing of AGV
        self.agv_pre_rout = {}
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
            self.agv_next_rout[chr(i + 65)] = (0, 0)
            self.agv_state[chr(i + 65)] = 0 # Initial state is 0
            self.agv_mode[chr(i + 65)] = 0 # Initial mode is normal
            self.agv_goal[chr(i + 65)] = [(0, 0), (0, 0), (0, 0), (0, 0)]
            self.agv_info[chr(i + 65)] = [0, 0]
            self.control_buffer[chr(i + 65)] = (0, 0)
            self.agv_rout[chr(i + 65)] = []
            self.agv_pre_rout[chr(i + 65)] = (0, 0)
        
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
            return self.agv_state[num]
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
            self.agv_pre_rout[num] = self.agv_pos[num]
            self.agv_next_rout[num] = self.agv_rout[num][0]
    
    def make_control(self):
        self.time += 1
        # Check there are probem.
        if self.running_opt == 0:
            # Make a routing for AGV
            self.dijkstra_rout()
        if self.running_opt == 1:
            self.DAA1()
        if self.running_opt == 2:
            self.DAA2()
        if self.running_opt == 3:
            pass
        return (self.control_buffer, self.agv_mode)
    
    # 수정 절대 금지!
    def dijkstra_rout(self):
        # Get the Dijkstra rout of AGVs
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]
        
            # Change the state of AGVs
            if (pos == goal):
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0
                self.agv_rout[num] = self.dijkstra_shortest(self.graph, pos, goal)
            
            # If AGV need next rout! (rout node)
            if (((self.map[pos[1]][pos[0]] == 6) or (pos in self.agv_goal[num])) and (self.agv_mode[num] == 0)):
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
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            next_pos = self.agv_next_pos[num]
            for num2 in self.agv_nums:
                if num != num2:
                    if (next_pos == self.agv_next_pos[num2]):
                        self.agv_mode[num] = 1
                        
            if self.map[pos[1]][pos[0]] == 1:
                self.agv_mode[num] = 2
                self.control_buffer[num] = (0, 0)
        
    # J Yoo et.al An algorithm for deadlock avoidance in an AGV System, 2005
    def DAA1(self):
        # Get the Dijkstra rout of AGVs
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]
        
            # Change the state of AGVs
            if (pos == goal):
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0
                self.agv_rout[num] = self.dijkstra_shortest(self.graph, pos, goal)
            
            # Deadlock occurs
            if ((self.agv_mode[num] == 1)):
                # If AGV locates in rout (node)
                if (pos == self.agv_pre_rout[num]): #or (pos == self.agv_next_rout[num])):
                    next_rout =  self.agv_rout[num][0]
                    self.insert_edge(num, (pos, next_rout))
                    if (len(self.agv_rout[num]) > 1):
                        next_next_rout = self.agv_rout[num][1]
                        self.insert_edge(num, (next_rout, next_next_rout))
                
                # AGV is in the zone (between the nodes)
                else:
                    # AGV can not enter this zone!
                    self.delete_edge((self.agv_pre_rout[num], self.agv_next_rout[num]), num)
                    # But AGV can enter the next zone
                    if (len(self.agv_rout[num]) > 1):
                        next_next_rout = self.agv_rout[num][1]
                        self.insert_edge(num, (next_rout, next_next_rout))
                
                # If next zone is idle
                now_zone = (self.agv_pre_rout[num], self.agv_next_rout[num])
                if (self.idle_zone(now_zone, num)):
                    if (self.zone_control(num, now_zone)):
                        #print(num, " need to find another way")
                        # if it has alternative routings
                        for neighbors in self.find_neighbors(now_zone[0][0], now_zone[0][1], False):
                            self.delete_edge(num, (now_zone[0], neighbors))
                        # Make Alternative pass
                        
                    else:
                        print(num, " has no problem")
                        
                # Zone is busy!
                else:
                    #print(num, "is busy at", now_zone)
                    # Wait!
                    self.control_buffer[num] = (0, 0)
            
            # Normal Mode
            if ((self.agv_mode[num] == 0)):
                # If AGV need next rout! (rout node)
                if (((pos == self.agv_next_rout[num]) or (pos in self.agv_goal[num]))):
                    self.out_zone((self.agv_pre_rout[num], self.agv_next_rout[num]))
                    #print(num, "out zone", (self.agv_pre_rout[num], self.agv_next_rout[num]))
                    next_rout = self.agv_rout[num][0]
                    
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
                    
                    # Transition of rout infomation
                    self.agv_pre_rout[num] = pos
                    self.agv_next_rout[num] = self.agv_rout[num][0]
                    #print("Make next rout of", num, "for" ,self.agv_next_rout[num] )
                
                # routing not changed
                else:
                    self.agv_next_pos[num] = (pos[0] + self.control_buffer[num][0], pos[1] + self.control_buffer[num][1]) 
                
        # Collision prevention => Deadlock
        for num in self.agv_nums:
            now_zone = (self.agv_pre_rout[num], self.agv_next_rout[num])
            #print(num, "now in ", now_zone)
            if (not self.idle_zone(now_zone, num)):
                self.agv_mode[num] = 1
                self.control_buffer[num] = (0, 0)
                continue
                        
            if self.map[self.agv_pos[num][1]][self.agv_pos[num][0]] == 1:
                self.agv_mode[num] = 2
                self.control_buffer[num] = (0, 0)
                continue
            
            for num2 in self.agv_nums:
                if num != num2:
                    if (self.agv_next_pos[num] == self.agv_next_pos[num2]):
                        self.agv_mode[num] = 1
                        self.control_buffer[num] = (0, 0)
                        break
            
            #print(num, "is now in", self.agv_pos[num], "with ", self.map[pos[1]][pos[0]])
            if (self.agv_pos[num] == self.agv_pre_rout[num]) and ((self.agv_mode[num] == 0)):
                # Everything is OKay, just go
                #print(num, "in zone", (self.agv_pre_rout[num], self.agv_next_rout[num]))
                self.in_zone((self.agv_pre_rout[num], self.agv_next_rout[num]), num)
                self.agv_rout[num].pop(0) 
                
        return 0
    
    # J Yoo et.al An algorithm for deadlock avoidance in an AGV System, 2005
    def DAA2(self):
        # Get the Dijkstra rout of AGVs
        for num in self.agv_nums:
            pos = self.agv_pos[num]
            state = self.agv_state[num]
            goal = self.agv_goal[num][state]
        
            # Change the state of AGVs
            if (pos == goal):
                state = self.change_state(num, state)
                goal = self.agv_goal[num][state]
                self.agv_mode[num] = 0
                self.agv_rout[num] = self.dijkstra_shortest(self.graph, pos, goal)
            
            # Deadlock occurs
            if ((self.agv_mode[num] == 1)):
                if (pos == self.agv_next_rout[num]):
                    print("Okay!")
                    self.agv_mode[num] = 0
                    self.agv_rout[num] = self.dijkstra_shortest(self.graph, pos, goal)
                
                # If AGV locates in rout (node)
                if (pos == self.agv_pre_rout[num]):
                    print(num, "is now in", pos, ", and going to ", self.agv_next_rout[num])
                    next_rout = self.alter_path(self.agv_pos[num], self.agv_next_rout[num])
                    print(num, "make alter path to", next_rout)
                    
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
                    
                    # Transition of rout infomation
                    self.agv_pre_rout[num] = pos
                    self.agv_next_rout[num] = next_rout
                    
                # AGV is in the zone (between the nodes)
                else:
                    pass
            
            # Normal Mode
            if ((self.agv_mode[num] == 0)):
                # If AGV need next rout! (rout node)
                if (((pos == self.agv_next_rout[num]) or (pos in self.agv_goal[num]))):
                    self.out_zone((self.agv_pre_rout[num], self.agv_next_rout[num]))
                    #print(num, "out zone", (self.agv_pre_rout[num], self.agv_next_rout[num]))
                    next_rout = self.agv_rout[num][0]
                    
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
                    
                    # Transition of rout infomation
                    self.agv_pre_rout[num] = pos
                    self.agv_next_rout[num] = self.agv_rout[num][0]
                    #print("Make next rout of", num, "for" ,self.agv_next_rout[num] )
                
                # routing not changed
                else:
                    self.agv_next_pos[num] = (pos[0] + self.control_buffer[num][0], pos[1] + self.control_buffer[num][1]) 
                
        # Collision prevention => Deadlock
        for num in self.agv_nums:
            now_zone = (self.agv_pre_rout[num], self.agv_next_rout[num])
            #print(num, "now in ", now_zone)
            if (not self.idle_zone(now_zone, num)):
                self.agv_mode[num] = 1
                self.control_buffer[num] = (0, 0)
                continue
                        
            if self.map[self.agv_pos[num][1]][self.agv_pos[num][0]] == 1:
                self.agv_mode[num] = 2
                self.control_buffer[num] = (0, 0)
                continue
            
            for num2 in self.agv_nums:
                if num != num2:
                    if (self.agv_next_pos[num] == self.agv_next_pos[num2]):
                        self.agv_mode[num] = 1
                        self.control_buffer[num] = (0, 0)
                        break
            
            #print(num, "is now in", self.agv_pos[num], "with ", self.map[pos[1]][pos[0]])
            if (self.agv_pos[num] == self.agv_pre_rout[num]) and ((self.agv_mode[num] == 0)):
                # Everything is OKay, just go
                #print(num, "in zone", (self.agv_pre_rout[num], self.agv_next_rout[num]))
                self.in_zone((self.agv_pre_rout[num], self.agv_next_rout[num]), num)
                self.agv_rout[num].pop(0) 
                
        return 0
    
    
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