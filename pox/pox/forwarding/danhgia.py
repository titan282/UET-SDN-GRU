from time import sleep

normal = 0 # True Positive
attack = 0 # False Positive

for i in range(500):
    with open('result.txt', 'r') as t2:
        tmp = t2.readline()
        res = tmp[0]
        if (res=='0'):
            normal += 1
        else:
            attack += 1

    print("normal: ", normal)
    print("attack: ", attack)
    sleep(5)