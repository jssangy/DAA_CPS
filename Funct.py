import random

class Color_dict():
    
    dic = {}
    
    def __init__(self, agv_num):
        for i in range (agv_num):
            self.dic[chr(i + 65)] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
def get_distance(pos1, pos2):
    x = abs(pos1[0] - pos2[0])
    y = abs(pos1[1] - pos2[1])
    if x == 0:
        return y
    if y == 0:
        return x