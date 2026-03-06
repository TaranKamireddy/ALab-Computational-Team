import serial
import time

PORT = "COM9"
BAUD = 115200

ser = serial.Serial(PORT, BAUD)

rpm_file = open("rpm_log.txt", "w")
hall_file = open("hall_log.txt", "w")

rpm_file.write("time,rpm\n")
hall_file.write("time,state\n")

startTime = time.time()

print("Logging started...")

while True:

    line = ser.readline().decode().strip()

    if not line:
        continue

    try:
        rpm, state = line.split(",")

        t = time.time() - startTime

        rpm_file.write(f"{t},{rpm}\n")
        hall_file.write(f"{t},{state}\n")

        rpm_file.flush()
        hall_file.flush()

        print("RPM:", rpm)

    except:
        pass