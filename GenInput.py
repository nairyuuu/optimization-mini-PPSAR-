import random as rd
import os

def test_gen(N, M, K, file):
    Q = [rd.randint(50,100) for _ in range(K)]
    Q_min = min(Q)
    q = [rd.randint(Q_min//5, Q_min//2) for _ in range(M)]

    distance_matrix = []

    num_Node = 2*N + 2*M + 1
    # making distance matrix
    for i in range(num_Node):
        distance_matrix.append([])
        for j in range(i + 1):
            if i == j:
                distance_matrix[i].append(0)
            else:
                distance = rd.randint(num_Node//10, num_Node)
                distance_matrix[i].append(distance)
                distance_matrix[j].append(distance)
    with open(file, 'w') as f:
        f.write(f"{M} {N} {K}\n")
        for _q in q:
            f.write(f'{_q} ')
        f.write(f'\n')
        for _Q in Q:
            f.write(f'{_Q} ')
        f.write(f'\n')
        for i in range(num_Node):
            for j in range(num_Node):
                f.write(f'{distance_matrix[i][j]} ')
            f.write(f'\n')

if __name__ == "__main__":
    s = 0
    for filename in os.getcwd():
        if filename.startswith("test"):
            i = int(filename[4:-4])
            s = max(i, s)

    while True:
        try:
            N, M, K = map(int, input().split())
            test_gen(N, M, K, f'.\\test case\\test{s}.txt')
            s += 1
            print(f'test{s}.txt')
        except:
            break