from smbus import SMBus
from time import sleep
import sys

if (sys.argv[2] == "forward"):
    value = 20
elif (sys.argv[2] == "backward"):
    value = -20
else:
    value = 0

tmotor = SMBus(5)
tmotor.write_i2c_block_data(0x01, 0x48, [0xaa])
tmotor.write_i2c_block_data(0x01, 0x46, [value])
sleep(float(sys.argv[1]))
tmotor.write_i2c_block_data(0x01, 0x46, [0])
