from smbus import SMBus
tservo = SMBus(6)
tservo.write_i2c_block_data(0x01, 0x48, [0xaa])
tservo.write_i2c_block_data(0x01, 0x43, [0])