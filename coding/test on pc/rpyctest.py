#!/usr/bin/python
import rpyc
import time

conn = rpyc.classic.connect('ev3dev')
ev3 = conn.modules['ev3dev.ev3']      # import ev3dev.ev3 remotely
ev3dev2 = conn.modules['ev3dev2.display']
core = conn.modules['ev3dev.core']
smbus = conn.modules['smbus']
subprocess = conn.modules['subprocess']

remote = conn.builtin.open("opencv_frame_0.png")
rpyc.classic.download(conn, "opencv_frame_0.png", "opencv_frame_0.png")
#conn = rpyc.classic.connect('ev3dev') # host name or IP address of the EV3
#ev3 = conn.modules['ev3dev.core.Sound']      # import ev3dev.ev3 remotely

#ound.set_volume(100)
core.Sound.play("test on pc/caojibai.wav")
subprocess.call("python3 test/test2.py",shell=True)
subprocess.call("python 'test on pc/imgtransferrpyc.py'",shell=True)

screen = ev3.Screen()
display = ev3dev2.display.Display()
image = conn.modules['PIL']
picture = image.Image.open("opencv_frame_0.png")
screen.image.paste(picture, (0, 0))
screen.update()
sleep(5)
#test
#motorB = ev3.LargeMotor('outB')
#motorB.run_to_rel_pos(position_sp=360, speed_sp=900, stop_action="hold")

#register arm
motorA = ev3.MediumMotor('outA')
#close
motorD.run_timed(time_sp=2000, speed_sp=-1000)
time.sleep(2)
#open
motorA.run_timed(time_sp=2000, speed_sp=1000)

#register servo
tservo = smbus.SMBus(4)
tservo.write_i2c_block_data(0x01, 0x48, [0xaa])
#arm go up
tservo.write_i2c_block_data(0x01, 0x42, [0])
#arm go down
tservo.write_i2c_block_data(0x01, 0x42, [90])
#arm turn 1
tservo.write_i2c_block_data(0x01, 0x43, [0])
#arm turn 2
tservo.write_i2c_block_data(0x01, 0x43, [255])

#register motor
tmotor = smbus.SMBus(3)
tmotor.write_i2c_block_data(0x01, 0x48, [0xaa])
tmotor.write_i2c_block_data(0x01, 0x45, [20])
tmotor.write_i2c_block_data(0x01, 0x46, [20])
