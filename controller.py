import serial

ser = serial.Serial('/dev/ttyUSB0', 9600)

i = 0

while True:
    write_line(i)
    print(ser.readline())
    i += 1
