import numpy as np


def read_base_matrix(filename):
    with open(filename) as f:
        base_matrix = f.readlines()
        for i in range(len(base_matrix)):
            base_matrix[i] = base_matrix[i].split()
            for j in range(len(base_matrix[i])):
                base_matrix[i][j] = int(base_matrix[i][j])
    return base_matrix

def right_shift(matrix: list, shift: int):
    shifted_matrix = []
    for i in range(len(matrix)):
        shifted_matrix.append(matrix[i][-shift:] + matrix[i][:-shift])
    return shifted_matrix


def to_circulant(size, n): # size - sie of a square matrix, n - number of shift in identity matrix
    # return zero matrix if n == -1
    matrix = [[0 for i in range(size)] for j in range(size)]
    if n == -1:
        return matrix
    for i in range(size):
        matrix[i][i] = 1
    return right_shift(matrix, n)
    


def from_base_to_standard(base_matrix, z):
    s = []
    for i in range(len(base_matrix)):
        s.append([])
        for j in range(len(base_matrix[0])):
            s[i].append(np.array(to_circulant(z, base_matrix[i][j])))
        s[i] = np.concatenate(s[i], axis=1)
    return np.concatenate(s)


def gdbf(y, check_matrix, max_it, L, alpha, rho, delta):

    m = np.size(check_matrix, 0)
    n = np.size(check_matrix, 1)

    x = np.sign(y)
    print("x = ", x)
    l = [0 for i in range(len(y))]
    x_ = np.array([(-el + 1) // 2 for el in x])
    for it in range(max_it):
        c = [1 for ii in range(m)]
        for j in range(m):
            p = 1
            for i in range(n):
                if check_matrix[j][i] != 0:
                    p *= x[i]
            c[j] = p
        
        # c = (x_ @ check_matrix.T) % 2
        print("correct coords number:", np.sum(c), "/", m)
        # print("c = ", c)
        if np.sum(c) == m:
            return x, it

        # if np.sum(c) == 0:
            # return x, it

        e = [0 for i in range(n)]
        for j in range(n):
            l[j] = min(l[j], L) + 1

            # sum computation
            summa = 0
            for k in range(m):
                if check_matrix[k][j] != 0:
                    # summa += -2 * c[k] + 1
                    summa += c[k]

            e[j] = alpha * x[j] * y[j] + summa + rho[l[j]]
        e_th = min(e) + delta

        for j in range(n):
            if e[j] < e_th:
                x[j] = -x[j]
                l[j] = 0      
    return x, it


def gdbf_exp():
    wer_ = 0
    noise = np.random.normal(0, dev, n)

    noised_codeword = np.ones(n) + noise
    print(n - np.sum(np.sign(noised_codeword)), n)
    x, it = gdbf(
        y=noised_codeword,
        check_matrix=h, 
        max_it=settings.MAX_ITERATIONS, 
        L=L, 
        alpha=settings.ALPHA,  
        rho=momentum, 
        delta=settings.DELTA
    )
    if np.sum(np.sign(x)) != n:
        wer_ += 1
        total_wer += 1
    # av += (n - np.sum(np.sign(x))) / 2 / n
    print(exp, time.time() - t1, wer_ / n, it)
    print('====')
    with open(f"errors_snr={round(snr, 2)}.txt", 'a') as f:
        f.write(f'error made: {n - np.sum(np.sign(noised_codeword))}; code length: {n}\n')
        f.write(f'time: {time.time() - t1}\nber: {(n - np.sum(np.sign(x))) / n}\nsnr: {snr}\ndev: {dev}\niterations: {it}\n\n')
    t1 = time.time()


if __name__ == '__main__':
    print(from_base_to_standard([
        [1,0, 0, 0],
        [-1, 2, 1, 1]        
        ], 3))