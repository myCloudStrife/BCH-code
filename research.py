import gf, bch, time
import numpy as np
import matplotlib.pyplot as plt

def plotCodeSpeed():
    plt.figure()
    ax = plt.axes()
    ax.set_xticks(np.arange(0, 32, 2))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    for q in range (2, 7):
        n = (1 << q) - 1
        x = list()
        y = list()
        for t in range (0, (n - 1 >> 1) + 1):
            code = bch.BCH(n, t)
            x.append(t)
            y.append((code.pm.shape[0] - code.g.size + 1.0) / n)
        plt.plot(x, y, label = 'n = ' + str(n))
    plt.grid()
    plt.legend()
    plt.title('Code speed')
    plt.xlabel('t')
    plt.ylabel('Speed (k/n)')
    plt.show()
    
def plotCodeDistance():
    plt.figure()
    ax = plt.axes()
    ax.set_xticks(np.arange(0, 32, 2))
    ax.set_yticks(np.arange(0, 64, 4))
    x = [t for t in range(0, 32)]
    y = [2 * t + 1 for t in range(0, 32)]
    plt.plot(x, y, label = '2t + 1')
    ranges = np.array([[0, 4], [1, 8], [5, 16], [13, 32]])
    for q in range (3, 7):
        n = (1 << q) - 1
        x = list()
        y = list()
        for t in range (ranges[q - 3][0], ranges[q - 3][1]):
            print(n, t)
            code = bch.BCH(n, t)
            x.append(t)
            y.append(code.dist())
        plt.plot(x, y, label = 'n = ' + str(n))
    plt.grid()
    plt.legend()
    plt.title('Code distance')
    plt.xlabel('t')
    plt.ylabel('Distance')
    plt.show()
    
def plotDecodeRate(n, t):
    plt.figure()
    ax = plt.axes()
    ax.set_xticks(np.arange(1, n + 1, max(n + 1 >> 4, 1)))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.grid(True, zorder = 1)
    x = [i for i in range(1, n + 1)]
    y = np.array(test(n, t, x))
    plt.bar(x, y[:, 0] + y[:, 1] + y[:, 2], width = 0.5, zorder = 2, label = 'correct')
    plt.bar(x, y[:, 1] + y[:, 2], width = 0.5, zorder = 2, label = 'detected')
    plt.bar(x, y[:, 2], width = 0.5, zorder = 2, label = 'wrong')
    plt.legend(loc = 'upper right')
    plt.title('BCH-code(' + str(n) + ', ' + str(t) + ')')
    plt.xlabel('Error number')
    plt.ylabel('Ratio')
    plt.show()
    
    
def codeTime(code_params, errors):
    euclid_time = list()
    pgz_time = list()
    for i, params in enumerate(code_params):
        print(params)
        code = bch.BCH(params[0], params[1])
        U = np.random.randint(0, 2, (100, params[0] - code.g.size + 1))
        W = code.encode(U)
        W = noise(W, errors[i])
        start_time = time.process_time()
        code.decode(W, 'euclid')
        euclid_time.append(time.process_time() - start_time)
        start_time = time.process_time()
        code.decode(W, 'pgz')
        pgz_time.append(time.process_time() - start_time)
    return euclid_time, pgz_time
    
def noise(A, n):
    B = np.copy(A)
    arr = np.zeros(A.shape[1], int)
    for i in range(n):
        arr[i] = 1
    for i in range(A.shape[0]):
        np.random.shuffle(arr)
        B[i] ^= arr
    return B

def test(n, t, errors):
    result_euc = list()
    #result_pgz = list()
    code = bch.BCH(n, t)
    polynom = np.zeros(n + 1, int)
    polynom[0] = 1
    polynom[-1] = 1
    if gf.polydiv(polynom, code.g, code.pm)[1] != []:
        print('Error!!!')
    U = np.random.randint(0, 2, (100, n - code.g.size + 1))
    V = code.encode(U)
    for v in V:
        if gf.polydiv(v, code.g, code.pm)[1] != []:
            print('Error!!!')
        if gf.polyval(v, code.R, code.pm).any():
            print('Error!!!')
    for k in code.g:
        if k != 0 and k != 1:
            print('Error!!!')
    for err in errors:
        correct = 0
        wrong = 0
        detected = 0
        W = noise(V, err)
        Vx = code.decode(W, 'euclid')
        for i in range(V.shape[0]):
            if np.array_equal(V[i], Vx[i]):
                correct += 1
            elif not np.array_equal(Vx[i], Vx[i]):
                detected += 1
            else:
                wrong += 1
        print(correct, detected, wrong)
        result_euc.append([correct / 100, detected / 100, wrong / 100])
        '''correct = 0
        wrong = 0
        detected = 0
        Vx = code.decode(W, 'pgz')
        for i in range(V.shape[0]):
            if np.array_equal(V[i], Vx[i]):
                correct += 1
            elif not np.array_equal(Vx[i], Vx[i]):
                detected += 1
            else:
                wrong += 1
        print(correct, detected, wrong)
        result_pgz.append((correct / 100, detected / 100, wrong / 100))'''
    return result_euc #, result_pgz
    
#plotCodeSpeed()

#plotCodeDistance()

#print(codeTime([[7, 1], [15, 1], [31, 1], [63, 1], [127, 1]], [1, 1, 1, 1, 1]))
#print(codeTime([[7, 2], [15, 3], [31, 3], [63, 5], [127, 7]], [1, 1, 1, 1, 1]))
#print(codeTime([[7, 3], [15, 7], [31, 7], [63, 11], [127, 15]], [1, 1, 1, 1, 1]))
#print(codeTime([[7, 2], [15, 3], [31, 3], [63, 5], [127, 7]], [2, 3, 3, 5, 7]))
#print(codeTime([[7, 3], [15, 7], [31, 7], [63, 11], [127, 15]], [3, 7, 7, 11, 15]))

#print(test(31, 5, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]))

#plotDecodeRate(7, 1)
#plotDecodeRate(15, 3)
#plotDecodeRate(15, 5)
#plotDecodeRate(31, 5)
