from pyfirmata import *
import time

board = Arduino('COM3')

escPin = 9
board.digital[escPin].mode = SERVO

def servoToPulse(servo):
  servo = max(0, min(180, servo))
  return int(1000 + (servo / 180.0) * 1000)

def pulseToServo(pulse):
  return int((pulse - 1000) * 180 / (2000 - 1000))

def move(servo):
  board.digital[escPin].write(servo)
  time.sleep(2)

def main():
  move(0)
  move(180)
    
if __name__ == "__main__":
  main()