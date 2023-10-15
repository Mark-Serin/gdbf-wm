from gdbf_tools import gdbf, from_base_to_standard, read_base_matrix
import numpy as np
import time
import math
import settings
import sys


# проверить, та ли матрица, не трансопнируем ли мы её, посмотреть, мб в ноутбуке правильная матрица

base_matrix = read_base_matrix(settings.BASE_MATRIX_FILE)
h = from_base_to_standard(base_matrix, 54)
n = np.size(h, 1)
m = np.size(h, 0)
print(f'Code length:{n}\nCode dimension:{m}')

L = len(settings.MOMENTUM) - 1

snr = 10

t1 = time.time()
total_time = time.time()

dev = math.sqrt(1 / 10**(snr / 10))
print(f'dev:{dev}')
av = 0

total_wer = 0

for exp in range(settings.EXP_NUMBER):
    wer_ = 0
    noise = np.random.normal(0, dev, n)

    noised_codeword = np.ones(n) + noise
    print("sum = ", np.sum(np.sign(noised_codeword)))
    print("errors:", n - np.sum(np.sign(noised_codeword)))
    start_exp = time.time()
    x, it = gdbf(
        y=noised_codeword,
        check_matrix=h,
        max_it=settings.MAX_ITERATIONS, 
        L=L, 
        alpha=settings.ALPHA,  
        rho=settings.MOMENTUM, 
        delta=settings.DELTA
    )
    end_exp = time.time()
    print(f'Experiment time: {end_exp - start_exp}')
    if np.sum(np.sign(x)) != n:
        np.savetxt(f'codeword_exp={exp}_snr={snr}', noised_codeword)
        wer_ += 1
        total_wer += 1
    print(f'Exp number: {exp}\nWER: {wer_ / n}\nTotal iterations number: {it}')
    with open(f"errors_snr={round(snr, 2)}.txt", 'a') as f:
        f.write(f'error made: {n - np.sum(np.sign(noised_codeword))}; code length: {n}\n')
        f.write(f'time: {time.time() - t1}\nber: {(n - np.sum(np.sign(x))) / n}\nsnr: {snr}\ndev: {dev}\niterations: {it}\n\n')
    t1 = time.time()
with open(f"errors_snr={round(snr, 2)}.txt", 'a') as f:
    f.write(f'total time: {time.time() - total_time}\ntotal wer: {total_wer / settings.EXP_NUMBER}\n')
    f.write('\n=========================\n\n')