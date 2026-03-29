import serial
import time

# PORT = "COM9"
PORT = "/dev/cu.usbmodem11201"
BAUD = 115200

ser = serial.Serial(PORT, BAUD)
hall_file = open("hall_log.txt", "w")


startTime = time.time()

print("Logging started...")

while True:
    print((a := ser.readline()))
    line = a.decode().strip()

    if not line:
        continue

    try:
        arduino_us, state = line.split(",")

        hall_file.write(f"{int(arduino_us)},{int(state)}\n")
        hall_file.flush()

    except:
        pass
    
    