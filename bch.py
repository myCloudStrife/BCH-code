import gf
import numpy as np

class BCH(object):
    
    def __init__(self, n, t):
        file = open('primpoly.txt', 'r')
        self.pm = np.empty(0, int)
        for line in file:
            for val in line.split(','):
                val = int(val)
                if (val > n):
                    self.pm = gf.gen_pow_matrix(val)
                    break
            if self.pm.size:
                break
        self.R = self.pm[:t * 2, 1]
        self.g = gf.minpoly(self.R, self.pm)[0]
        file.close()
        return
    
    def encode(self, U):
        k = U.shape[1]
        n = self.pm.shape[0]
        result = np.empty((U.shape[0], n), int)
        for i, u in enumerate(U):
            result[i] = np.hstack((u, np.zeros(n - k)))
            result[i] = gf.polyadd(result[i], gf.polydiv(result[i], self.g, self.pm)[1])
        return result
    
    def decode(self, W, method = 'euclid'):
        result = np.empty(W.shape)
        for index, w in enumerate(W):
            result[index] = w
            s = gf.polyval(w, self.R, self.pm)
            if np.all(s == 0):
                continue
            if method == 'euclid':
                s = np.hstack((s[::-1], np.array([1], int)))
                i = 0
                while s[i] == 0:
                    i += 1
                loc = gf.euclid(np.hstack((np.array([1], int), np.zeros(self.R.size + 1, int))),
                                s[i:], self.pm, self.R.size >> 1)[2]
            else:
                t = self.R.size >> 1
                A = np.empty((t, t), int)
                for i in range(t):
                    A[i] = s[i:i + t]
                for v in range(t, 0, -1):
                    loc = gf.linsolve(A[:v, :v], s[v:2 * v], self.pm)
                    if np.array_equal(loc, loc):
                        loc = np.hstack((loc, np.array([1], int)))
                        break                        
                if not np.array_equal(loc, loc):
                    result[index].fill(np.nan)
                    continue
            roots_count = 0
            for i, val in enumerate(self.pm[:,1]):
                if gf.polyval(loc, val.reshape(1), self.pm)[0] == 0:
                    roots_count += 1
                    result[index][i] = int(result[index][i]) ^ 1
            if method == 'euclid' and roots_count != loc.size - 1:
                result[index].fill(np.nan)
            elif method == 'pgz':
                s = gf.polyval(result[index].astype(int), self.R, self.pm)
                if np.any(s != 0):
                    result[index].fill(np.nan)
        return result
    
    def dist(self):
        u = np.empty(self.pm.shape[0] - self.g.size + 1, int)
        distance = self.pm.shape[0]
        for i in range(1, 1 << u.size):
            for j in range(u.size):
                u[-1 - j] = (i >> j) & 1
            count = (self.encode(u.reshape(1, u.size))[0] == 1).sum()
            if distance > count:
                distance = count
        return distance