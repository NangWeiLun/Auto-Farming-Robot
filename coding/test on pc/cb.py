#!/usr/bin/python3
from smbus import SMBus
from time import sleep
tservo = SMBus(6)
tservo.write_i2c_block_data(0x01, 0x48, [0xaa])
tservo.write_i2c_block_data(0x01, 0x42, [0])
sleep(0.5)
tservo.write_i2c_block_data(0x01, 0x42, [55])
sleep(0.5)