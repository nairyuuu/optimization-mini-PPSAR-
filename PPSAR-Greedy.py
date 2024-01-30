import sys
import time
def import_data():
    global N, M, K, demands, taxi_capacity, routes, taxi_dist, visited, distance_matrix

    N, M, K = map(int, input().split())

    demands = [0 for _ in range(2 * N + 2 * M + 1)]

    demands[1 + N:1 + N + M] = list(map(int, input().split()))

    for i in range(1, M + 1):
        demands[i + 2 * N + M] = -demands[i + N]

    taxi_capacity = [0] + list(map(int, input().split()))
    routes = [[0] for _ in range(K + 1)]
    taxi_dist = [0 for _ in range(K + 1)]

    visited = [False for _ in range(2 * N + 2 * M + 1)]

    distance_matrix = [[] for _ in range(2 * N + 2 * M + 1)]

    for i in range(2 * N + 2 * M + 1):
        distance_matrix[i] = list(map(int, input().split()))

    visited[0] = True

    visited[0] = True

def solve():
    global total_dist, current_dist, capacity, _lower_bound, routes, ans, taxi_dist
    current_capacity = [0 for _ in range(K + 1)]
    valid = [True for _ in range(K + 1)]

    num_visited = 0

    while num_visited < 2 * N + 2 * M:
        best_taxi = -1
        best_dist = float('inf')
        # chọn taxi có khoảng cách đã đi ngắn nhất
        for i in range(1, K + 1):
            if taxi_dist[i] < best_dist and valid[i]:
                best_dist = taxi_dist[i]
                best_taxi = i

        if best_taxi == -1:
            # print("here")
            break

        current_node = routes[best_taxi][-1]
        min_dist = float('inf')

        is_human = False
        best_node = -1
        # tìm điểm mà khi đi đến mà khoảng cách tăng thêm ít nhất
        for node in range(1, 2 * N + 2 * M + 1):
            distance = distance_matrix[current_node][node]
            # nếu đã đi qua hoặc vượt quá khả năng pickup của xe, tìm đến điểm tiếp theo
            if visited[node] or current_capacity[best_taxi] + demands[node] > taxi_capacity[best_taxi]:
                continue
            # nếu đây là điểm trả hàng
            if 1 + 2 * N + M <= node <= 2 * N + 2 * M:
                # nếu chưa được lấy hàng, tìm điểm tiếp theo
                if not visited[node - N - M]:
                    continue
                # nếu đã được lấy, kiểm tra xem nếu xe này có phải xe lấy không
                elif node - N - M not in routes[best_taxi]:
                    continue
            # nếu đây là điểm đến của passenger, chọn điểm tiếp theo vì xe luôn đi đến điểm đến sau khi đón khách
            if 1 + N + M <= node <= 2 * N + M:
                continue
            # nếu đây là điểm đi của passenger
            if 1 <= node <= N:
                # thêm khoảng cách với điểm đến
                distance += distance_matrix[node][node + N + M]
                if distance < min_dist:
                    min_dist = distance
                    is_human = True
                    best_node = node
            # nếu đây là điểm đi của của hàng hóa
            else:
                if distance < min_dist:
                    min_dist = distance
                    is_human = False
                    best_node = node


        # nếu không tìm được điểm phù hợp
        if best_node == -1:
            valid[best_taxi] = False
        # nếu đã chọn được điểm
        else:
            # nếu điểm đấy là passenger thì đi hẳn đến điểm đến của người đấy và cộng thêm khoảng cách, 2 điểm đã đi
            if is_human:
                routes[best_taxi].append(best_node)
                routes[best_taxi].append(best_node + N + M)
                taxi_dist[best_taxi] += min_dist
                visited[best_node] = True
                visited[best_node + N + M] = True
                num_visited += 2
            # nếu điểm đấy là hàng thì cộng thêm khoảng cách, 1 điểm đã đi
            else:
                routes[best_taxi].append(best_node)
                taxi_dist[best_taxi] += min_dist
                visited[best_node] = True
                num_visited += 1
                current_capacity[best_taxi] += demands[best_node]
            # reset valid
            valid = [True for _ in range(K + 1)]

def objective_value(routes):
    sum_route = []
    for i in range(1, K + 1):
        s = 0
        for j in routes[i]:
            s += distance_matrix[j-1][j]
        sum_route.append(s)
    print(max(sum_route))
def print_sol():
    print(K)
    for i in range(1, K + 1):
        print(len(routes[i]) + 1)
        print(" ".join(map(str, routes[i])), "0")

if __name__ == "__main__":
    # start_time = time.time()
    # s = 25
    # sys.stdin = open(f'test case\\test{s}.txt', "r")
    # sys.stdout = open(f'.\\result\greedy\\test{s}.txt', "w")
    import_data()
    solve()
    # objective_value(routes)
    print_sol()
    # end_time = time.time()
    # print(f'Running Time = {end_time - start_time} seconds')