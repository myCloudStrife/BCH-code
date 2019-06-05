import numpy as np

def gen_pow_matrix(primpoly):
    q = len(bin(primpoly)) - 3
    pm = np.empty(((1 << q) - 1, 2), int)
    alpha = 2
    for i in range(pm.shape[0]):
        pm[i][1] = alpha
        pm[alpha - 1][0] = i + 1
        alpha <<= 1
        if alpha > pm.shape[0]:
            alpha ^= primpoly
    return pm

def add(X, Y):
    return X ^ Y

def sum(X, axis = 0):
    result = np.zeros((X.shape[1 - axis]), int)
    for i in range(X.shape[axis]):
        if axis == 0:
            result ^= X[i]
        else:
            result ^= X[:, i]
    if axis == 0:
        return result.reshape(1, result.size)
    return result.reshape(result.size, 1)

def prod(X, Y, pm):
    result = np.zeros(X.shape, int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j] == 0 or Y[i][j] == 0:
                result[i][j] = 0
            else:
                result[i][j] = pm[(pm[X[i][j] - 1][0] + pm[Y[i][j] - 1][0] - 1) % pm.shape[0]][1]
    return result

def divide(X, Y, pm):
    result = np.zeros(X.shape, int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            degree = pm[X[i][j] - 1][0] - pm[Y[i][j] - 1][0] - 1
            if degree < 0:
                degree += pm.shape[0]
            result[i][j] = pm[degree][1]
            if X[i][j] == 0:
                result[i][j] = 0
            if Y[i][j] == 0:
                result[i][j] = np.nan
    return result

def linsolve(A, b, pm):
    SLAU = np.hstack((A, b.reshape(A.shape[0], 1)))
    for i in range(SLAU.shape[0]):
        if (SLAU[i][i] == 0):
            j = i + 1
            while j < SLAU.shape[0] and SLAU[j][i] == 0:
                j += 1
            if j == SLAU.shape[0]:
                return np.nan
            SLAU[i], SLAU[j] = SLAU[j], SLAU[i].copy()
        for j in range(i + 1, SLAU.shape[1]):
            SLAU[i][j] = divide(SLAU[i][j].reshape(1, 1), SLAU[i][i].reshape(1, 1), pm)
        SLAU[i][i] = 1
        for j in range(i + 1, SLAU.shape[0]):
            for k in range(i + 1, SLAU.shape[1]):
                SLAU[j][k] = add(SLAU[j][k],
                                 prod(SLAU[i][k].reshape(1, 1), SLAU[j][i].reshape(1, 1), pm))
            SLAU[j][i] = 0
    for i in range(SLAU.shape[0] - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            SLAU[j][-1] = add(SLAU[j][-1],
                              prod(SLAU[i][-1].reshape(1, 1), SLAU[j][i].reshape(1, 1), pm))
    return SLAU[:, -1]

def minpoly(x, pm):
    roots = list()
    for val in x:
        if val == 0:
            roots.append(val)
        while not (val in roots):
            roots.append(val)
            degree = pm[val - 1][0]
            degree <<= 1
            if (degree > pm.shape[0]):
                degree -= pm.shape[0]
            val = pm[degree - 1][1]
    roots.sort()
    roots = np.asarray(roots, int)
    polynom = np.zeros((1, len(roots) + 1),  int)
    polynom[0][-1] = 1
    for i in range(0, len(roots)):
        polynom = add(np.hstack((polynom[0][1:polynom.shape[1]], 0)),
                      prod(polynom, np.full((1, len(roots) + 1), roots[i], int), pm))
    return polynom.reshape(polynom.size), roots

def polyval(p, x, pm):
    result = np.zeros(len(x), int)
    for i, val in enumerate(x):
        mult = np.array([[1]], int)
        val = np.array([[val]], int)
        for k in reversed(p):
            result[i] = add(result[i], prod(mult, k.reshape(1, 1), pm))
            mult = prod(mult, val, pm)
    return result

def polyprod(p1, p2, pm):
    if len(p1) == 0 or len(p2) == 0:
        return np.empty(0, int)
    result = np.zeros(len(p1) + len(p2) - 1, int)
    p1_length = len(p1)
    for i, val in enumerate(p2):
        result[i:p1_length + i] ^= prod(p1.reshape(1, p1_length),
                                        np.full((1, p1_length), val, int),
                                        pm).reshape(p1_length)
    return result

def polydiv(p1, p2, pm):
    p1_degree = len(p1) - 1
    p2_degree = len(p2) - 1
    if p1_degree < p2_degree:
        return np.empty(0, int), p1
    q = divide(p1[0].reshape(1, 1), p2[0].reshape(1, 1), pm)
    r = add(p1, np.hstack((prod(p2.reshape(1, p2.size), np.full((1, p2.size), q, int), pm),
                                   np.zeros((1, p1_degree - p2_degree), int))))
    q = q.reshape(1)
    r = r.reshape(r.size)
    i = 0
    while i < r.size and r[i] == 0:
        i += 1
    if (i < r.size):
        first_part = np.zeros(min(i, p1_degree - p2_degree + 1), int)
        first_part[0] = q[0]
        q, r = polydiv(r[i:], p2, pm)
        return np.hstack((first_part, q)), r
    return np.hstack((q, np.zeros(p1_degree - p2_degree, int))), np.empty(0, int)

def polyadd(p1, p2):
    if len(p1) > len(p2):
        return add(p1, np.hstack((np.zeros(len(p1) - len(p2), int), p2)))
    return add(np.hstack((np.zeros(len(p2) - len(p1), int), p1)),  p2)

def euclid(p1, p2, pm, max_deg = 0):
    if len(p2) > len(p1):
        d, b, a = euclid(p2, p1, pm, max_deg)
        return d, a, b
    x2 = np.array([1], int)
    x1 = np.array([], int)
    y2 = np.array([], int)
    y1 = np.array([1], int)
    while len(p2) - 1 > max_deg:
        q, r = polydiv(p1, p2, pm)
        x = polyadd(x2, polyprod(x1, q, pm))
        y = polyadd(y2, polyprod(y1, q, pm))
        x2 = x1
        x1 = x
        y2 = y1
        y1 = y
        p1 = p2
        p2 = r
    if len(p2) == 0:
        return p1, x2, y2
    return p2, x1, y1