from smbus import SMBus
from time import sleep
import sys

tmotor = SMBus(5)
tmotor.write_i2c_block_data(0x01, 0x48, [0xaa])
if (sys.argv[3] == "fast"):
    speed = 6
elif (sys.argv[3] == "slow"):
    speed = 1

if (sys.argv[2] == "forward"):
    valuel = -15
    valuer = 15
elif (sys.argv[2] == "backward"):
    valuel = 15
    valuer = -15
elif (sys.argv[2] == "turnleft"):
    valuel = 15
    valuer = 15
elif (sys.argv[2] == "turnright"):
    valuel = -15
    valuer = -15
elif (sys.argv[2] == "swipeBright"):
    valuel = 15
    valuer = 0
elif (sys.argv[2] == "swipeBleft"):
    valuel = 0
    valuer = -15
elif (sys.argv[2] == "swipeFleft"):
    valuel = 0
    valuer = 15
elif (sys.argv[2] == "swipeFright"):
    valuel = -15
    valuer = 0

tmotor.write_i2c_block_data(0x01, 0x45, [valuel*speed])
tmotor.write_i2c_block_data(0x01, 0x46, [valuer*speed])
sleep(float(sys.argv[1]))
tmotor.write_i2c_block_data(0x01, 0x45, [0])
tmotor.write_i2c_block_data(0x01, 0x46, [0])
