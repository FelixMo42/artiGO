import random
import serial

ser = serial.Serial('/dev/ttyUSB0', 9600)

RS = 0
LS = 0

while True:
    msg = str(RS) + " " + str(LS)
    ser.write(msg.encode('utf-8'))
    sens = str(ser.readline())
    print(sens)
    RS = max(min(RS + random.randint(-50,50), 255), -255)
    LS = max(min(LS + random.randint(-50,50), 255), -255)
    print("RS: ", RS)
    print("LS: ", LS)
