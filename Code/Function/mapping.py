from queue import PriorityQueue
from ugot import Robot

# 0: Đường đi, 1: Chướng ngại vật
map_grid = [
    [0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0]
]

# Các điểm chỉ định
start_point = (0, 0)
end_point = (4, 4)

# Khởi tạo robot
robot = Robot()

# Kết nối với robot
robot.connect('your_robot_id_or_ip_address')

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(map_grid, start, end):
    rows, cols = len(map_grid), len(map_grid[0])
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and map_grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    open_set.put((f_score[neighbor], neighbor))
    return None

path = a_star(map_grid, start_point, end_point)
print("Đường đi:", path)

def move_robot(path):
    for point in path:
        # Gửi lệnh điều khiển tới robot dựa trên điểm tiếp theo trong đường đi
        # Ví dụ: robot.move_forward(), robot.turn_left(), etc.
        # Cần tính toán hướng di chuyển từ điểm hiện tại đến điểm tiếp theo
        pass

move_robot(path)
