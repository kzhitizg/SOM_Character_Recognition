import numpy as np
import random
import matplotlib.pyplot as plt

def print_pattern(arr):
    i=0
    for a in range(5):
        for b in range(5):
            if arr[0, i] == 1:
                print('*', end='')
            else:
                print(' ', end='')
            i+=1
        print()
    print()

def read_data(file:str):
    '''Read the numbers from file, each 5x4'''
    f = open(file, 'r')
    mat = []
    row = []
    while True:
        x = f.read(1)
        if not x:
            break
        if x == "-":
            row.append(0)
        elif x=="*":
            row.append(1)
        else:
            continue
        if len(row) == 25:
            mat.append(row)
            row=[]
    
    data = np.matrix(mat)
    return data

dataset = read_data("nums.txt")

rows, cols = dataset.shape

clusters = 3
alpha = 0.5
weights = np.random.rand(cols, clusters)
epoc = 1
thr = 0.001
R = 3

while epoc <=100:
    for i in range(rows):
        x = dataset[i]
        d = np.array([np.linalg.norm(x-weights[:, j]) for j in range(clusters)])
        bmu = min(d)
        if R<bmu:
            ind = np.where(d == bmu)[0]
            delta = alpha * (x-weights[:, ind].T)
            delta = np.array(delta)
            weights[:, ind] = weights[:, ind] + delta.T
        else:
            for ind in range(clusters):
                if d[ind]<R:
                    delta = alpha * (d[ind]/bmu) * (x-weights[:, ind].T)
                    delta = np.array(delta)
                    weights[:, ind] = weights[:, ind] + delta
    epoc+=1
    alpha/=2
    R/=2

print(f"Weights- {weights}")

res = [[] for _ in range(clusters)]
for i in range(rows):
    x = dataset[i]
    d = np.array([np.linalg.norm(x-weights[:, j]) for j in range(clusters)])
    ind = np.where(d == min(d))[0][0]
    res[ind].append(i)

for cl in res:
    print("Cluster= ")
    for patt in cl:
        print_pattern(dataset[patt])