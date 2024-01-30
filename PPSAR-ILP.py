from ortools.linear_solver import pywraplp
import time
import sys

def input_data(file_path):
    data = {}
    f = open(file_path, "r")
    
    N, M, K = map(int, f.readline().split())
    
    data["parcel_amount"] = [0 for i in range(N+1)]
    data["parcel_amount"].extend([int(q) for q in f.readline().split()])
    
    data["capacity"] = [0]
    data["capacity"].extend([int(Q) for Q in f.readline().split()])
    
    data["distance"] = []
    for i in range(2*N + 2*M + 1):
        data["distance"].append([int(d) for d in f.readline().split()])
        point_0 = data["distance"][i][0]
        data["distance"][i].append(point_0)
    data["distance"].append(data["distance"][0])
    return N, M, K, data

if __name__ == "__main__":
    
    #input
    
    s = 4
    file_path = f".\\test case\\test{s}.txt"
    # print(file_path)
    N, M, K, data = input_data(file_path)
    
    start_time = time.time()
    
    #Create solver
    solver = pywraplp.Solver.CreateSolver("SAT")
    
    alpha = 1000000
    
    #Create variable
    
    #X[i][j][k] = 1 if the k-th taxis move from i-th point to j-th point, otherwise X[i][j][k] = 0
    
    X = [[[0 for k in range(K + 1)] for j in  range(2*M + 2*N + 2)] for i in range(2*M + 2*N + 2)]
    
    for i in range(2*N + 2*M + 2):
        for j in range(2*N + 2*M + 2):
            if i != j:
                for k in range(1, K + 1):
                    X[i][j][k] = solver.IntVar(0, 1, 'X[%d][%d][%d]' % (i, j, k))
    
    #Y[k][j]: the amount of parcel in the k-th taxis after it leaves j-th point
    
    Y = [[0 for i in range(2*N + 2*M + 2)] for i in range(K + 1)]
                    
    for k in range(1, K + 1):
        for j in range(2*N + 2*M + 1):
            Y[k][j] = solver.IntVar(0, data["capacity"][k], 'Y[%d][%d]' % (k, j))
    
    #Z[k][j]: The order of j-th point in the route of k-th taxi
    Z = [[0 for i in range(2*N + 2*M + 2)] for k in range(K + 1)]
    
    for k in range(1, K + 1):
        for j in range(2*N + 2*M + 2):
            Z[k][j] = solver.IntVar(0, 2*N + 2*M + 1, 'Z[%d][%d]' % (k, j))
    
    #Add constraints
    
    #In_degree and out_degree of each point from 1 to 2*N + 2*M are equal to 1
    for i in range(1, 2*N + 2*M + 1):
        in_deg_i = []
        out_deg_i = []
        for j in range(2*N + 2*M + 2):
            if i != j:
                for k in range(1, K + 1):
                    in_deg_i.append(X[j][i][k])
                    out_deg_i.append(X[i][j][k])
        solver.Add(sum(in_deg_i) == 1)
        solver.Add(sum(out_deg_i) == 1)
        
    #For each taxi, in_degree and out_degree of each point from 1 to 2*N + 2*M are equal
    for k in range(1, K + 1):
        for i in range(1, 2*N + 2*M + 1):
            in_deg_i = []
            out_deg_i = []
            for j in range(2*N + 2*M + 2):
                if i != j:
                    in_deg_i.append(X[j][i][k])
                    out_deg_i.append(X[i][j][k])
            solver.Add(sum(in_deg_i) == sum(out_deg_i))
        
    #For each taxi, out_degree of start point and in_degree of end point are equal to 1
    for k in range(1, K + 1):
        out_deg_start = []
        in_deg_end = []
        for i in range(1, N + M + 1):
            out_deg_start.append(X[0][i][k])
        for i in range(N + M + 1, 2*N + 2*M + 1):
            in_deg_end.append(X[i][2*N + 2*M + 1][k])
        solver.Add(sum(out_deg_start) == 1)
        solver.Add(sum(in_deg_end) == 1)
        
    #For each taxi, in_degree  of start point and out_degree of end point are equal to 0
    for k in range(1, K + 1):
        in_deg_start = []
        out_deg_end = []
        for i in range(2*N + 2*M + 2):
            in_deg_start.append(X[i][0][k])
            out_deg_end.append(X[2*N + 2*M + 1][i][k])
        solver.Add(sum(in_deg_start) == 0)
        solver.Add(sum(out_deg_end) == 0)
        
    #If the k-th taxis pick up a passenger at i-th point, it must move to i+N+M-th point instantly
    #(No stopping point between i-th point and i+N+M-th point)
    for k in range (1, K + 1):
        for i in range(0, 2*N + 2*M + 1):
            for j in range(1, N+1):
                if i != j:
                    solver.Add(alpha*(1 - X[i][j][k]) + 1 >= X[j][j+N+M][k])
                    solver.Add(alpha*(1 - X[i][j][k]) + X[j][j+N+M][k] >= 1)
    
    #If the k-th taxi move from i-th point to j-th point (j in range N+1 - N+M) ==> Y[k][j] = Y[k][i] + data["parcel_amount"][j]
    #If the k-th taxi move from i-th point to j-th point (j in range 2N+M+1 - 2N+2M) ==> Y[k][j] = Y[k][i] - data["parcel_amount"][j - M - N]
    for k in range(1, K + 1):
        for i in range(0, 2*N + 2*M + 1):
            for j in range(N + 1, N + M + 1):
                if i != j:
                    solver.Add(alpha*(1 - X[i][j][k]) + Y[k][j] >= Y[k][i] + data["parcel_amount"][j])
                    solver.Add(alpha*(1 - X[i][j][k]) + Y[k][i] + data["parcel_amount"][j] >= Y[k][j])

            for j in range(2*N + M + 1, 2*N + 2*M + 1):
                if i != j:
                    solver.Add(alpha*(1 - X[i][j][k]) + Y[k][j] >= Y[k][i] - data["parcel_amount"][j - M - N])
                    solver.Add(alpha*(1 - X[i][j][k]) + Y[k][i] - data["parcel_amount"][j - M - N] >= Y[k][j])
                    
    #For each taxi, out_degree of i-th point and in_degree of i+N+M-th point are equal
    for k in range(1, K + 1):
        for i in range(N + 1, N+M+1):
            out_deg_i = []
            in_deg_iNM = []
            for j in range(1, 2*N + 2*M + 1):
                if i != j:
                    out_deg_i.append(X[i][j][k])
                if j != i + N + M:
                    in_deg_iNM.append(X[j][i+N+M][k])
        
            solver.Add(sum(out_deg_i) == sum(in_deg_iNM))
    
    for k in range(1, K + 1):
        solver.Add(Y[k][0] == 0)
        
    for k in range(1, K + 1):
        solver.Add(Z[k][0] == 0)
            
    for k in range(1, K + 1):
        for i in range(2*N + 2*N + 1):
            for j in range(1, 2*N + 2*N + 2):
                solver.Add(alpha * (1 - X[i][j][k]) + Z[k][j] >= Z[k][i] + 1)
                solver.Add(alpha * (1 - X[i][j][k]) + Z[k][i] + 1 >= Z[k][j])
                
    for k in range(1, K + 1):
        for i in range(1, N + M + 1):
            solver.Add(Z[k][i] <= Z[k][i + N + M])
            
    #Set the 1st route is the longest route
    lengthOfRoute = [0]
    for k in range(1, K + 1):
        constrain_expr = []
        for i in range(2*N + 2*M + 2):
            for j in range(2*N + 2*M + 2):
                if i != j:
                    constrain_expr.append(X[i][j][k] * data["distance"][i][j])
        lengthOfRoute.append(sum(constrain_expr))
    
    for k in range(2, K + 1):
        solver.Add(lengthOfRoute[1] >= lengthOfRoute[k])
    
    #Minimize the length of 1st route              
    solver.Minimize(lengthOfRoute[1])
    status = solver.Solve()
    
    #Print output
    
    ou_file = f"result\ilp\\test{s}.txt"

    # sys.stdout = open(ou_file, "w")
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f'Objective value: {solver.Objective().Value()}')
        for k in range(1, K + 1):
            path = [0]
            length = 0
            while path[-1] != 2*N + 2*M + 1:
                for i in range(2*N + 2*M + 2):
                    if i != path[-1]:
                        if X[path[-1]][i][k].solution_value() == 1:
                            path.append(i)
                            length += data["distance"][path[-2]][i]
                            break
            print(f"Xe {k}:", ' --> '.join(map(str, path)), ', length =', length, ',count =', len(path))
                
    end_time = time.time()
                
    print(f'Running time        : {end_time - start_time} seconds' )
        
    sys.stdout = sys.__stdout__