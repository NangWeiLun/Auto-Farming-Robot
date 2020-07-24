from smbus import SMBus

tservo = SMBus(6)
tservo.write_i2c_block_data(0x01, 0x48, [0xaa])
tservo.write_i2c_block_data(0x01, 0x42, [0])

from ev3dev.ev3 import *
from time import sleep

m = MediumMotor('outD')
m.run_to_rel_pos(position_sp=360, speed_sp=900, stop_action="hold")

from ev3dev2.motor import MediumMotor
from ev3dev2.motor import SpeedDPS, SpeedRPM, SpeedRPS, SpeedDPM

mm = MediumMotor('outD')
mm.on_for_seconds(50,3)
mm.on_for_rotations(50,3)
mm.on_for_rotations(-50,2)
mm.on_for_rotations(50,2)

from ev3dev import ev3
ultraS = ev3.UltrasonicSensor('in1') #register ultrasonic
ultraS.mode='US-DIST-CM'
ultraS.value()

tmotor.write_i2c_block_data(0x01, 0x45, [-20])
tmotor.write_i2c_block_data(0x01, 0x46, [20])
sleep(1)
tmotor.write_i2c_block_data(0x01, 0x45, [0])
tmotor.write_i2c_block_data(0x01, 0x46, [0])


motorD.run_timed(time_sp=750, speed_sp=1000)
subprocess.call("python 'test on pc/armgoup.py'",shell=True)#arm go up
motorD.run_timed(time_sp=750, speed_sp=-1000)
subprocess.call("python 'test on pc/armgodown.py'",shell=True)#arm go down
