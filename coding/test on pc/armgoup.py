#!/usr/bin/python3
from smbus import SMBus
from time import sleep
tservo = SMBus(6)
tservo.write_i2c_block_data(0x01, 0x48, [0xaa])
tservo.write_i2c_block_data(0x01, 0x42, [35])
for i in range(35,0,-2):
    tservo.write_i2c_block_data(0x01, 0x42, [i])
    sleep(0.2)