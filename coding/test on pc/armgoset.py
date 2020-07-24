#!/usr/bin/python3
from smbus import SMBus
from time import sleep
import sys
tservo = SMBus(6)
tservo.write_i2c_block_data(0x01, 0x48, [0xaa])
value1 = int(sys.argv[1])
value2 = int(sys.argv[2])
value3 = 0
if(value1 < value2):
    value3 = 5
else:
    value3 = -5
for i in range(value1,value2,value3):
    tservo.write_i2c_block_data(0x01, 0x42, [i])
    value4 = abs(i-110)
    if(value4 < 40):
        value4 = 40
    elif(value4 > 110):
        value4 = 110
    tservo.write_i2c_block_data(0x01, 0x43, [value4])
    sleep(0.1)