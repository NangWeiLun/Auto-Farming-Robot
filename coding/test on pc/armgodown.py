#!/usr/bin/python3
from smbus import SMBus
from time import sleep
tservo = SMBus(6)
tservo.write_i2c_block_data(0x01, 0x48, [0xaa])
for i in range(0,35,2):
    tservo.write_i2c_block_data(0x01, 0x42, [i])
    sleep(0.2)
tservo.write_i2c_block_data(0x01, 0x42, [45])