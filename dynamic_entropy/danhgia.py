from time import sleep
import time

n_normal = 0 # True Positive
n_attack = 0 # False Positive
a_normal = 0
a_attack = 0

for i in range(1,500):
    start = time.time()
    with open('result.txt', 'r') as t1:
        tmp = t1.readline()
        res = tmp[0]
    with open('hihi.txt', 'r') as t3:
        tmp1 = t3.readline()
        flag = tmp1[0]
        if (flag == '1'):
            if (res == '0'):
                a_normal += 1
            else:
                a_attack += 1    
        else:
            if (res == '1'):
                n_attack += 1
            else:
                n_normal += 1
    print(flag,' ', res)
    print("a_count lan ", i)
    print("n_normal: ", n_normal)
    print("n_attack: ", n_attack)
    print("a_normal: ", a_normal)
    print("a_attack: ", a_attack)
    end = time.time()
    sleep(3-(end-start))