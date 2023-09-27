import os
import time
from detection import Entropy
import numpy as numpy
import datetime
import sys

diction = {}
timerSet =False
#port = sys.argv[1]

# Gan doi tuong cho class Entropy da dinh nghia tu file detectionUsingEntropy.py

aveEntropy = []	######    
i = 1
flag = 0

while(1):
  start = time.time()
  ent_obj = Entropy()
  ent_obj.start() #.#

  aveEntropy.append(ent_obj.value)
  # Neu cac list co >= 10 phan tu thi xoa het phan tu, tro lai list rong
  if len(aveEntropy) == 15:	######
      del aveEntropy[:]	######


  if ent_obj.value < 1.03:  #25%a 1.03 #50%a 0.74 #75%a 0.48
    flag += 1
  else:
    flag = 0

  print("Tinh cac gia tri voi lan collect ", i)
  print ("Entropy: ", str(ent_obj.value))
  print("                                   ")

  with open('thongke2_entropy.txt', 'a') as f:
    f.write(str(ent_obj.value) + "\n")

  # # So sanh neu gia tri entropy tuc thoi < nguong entropy dong
  if flag == 5:	######
    with open('result.txt', 'w') as f:
                f.write("1")
  else :
    with open('result.txt', 'w') as f:
                f.write("0")
  
  end = time.time()
  x = 2 - (end - start)
  #print(x)
  i+=1   
  time.sleep(x)