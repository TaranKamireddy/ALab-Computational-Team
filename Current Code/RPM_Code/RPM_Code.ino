#define HALL_A 2
#define HALL_B 3
#define HALL_C 4

#define POLE_COUNT 4 
#define HALL_COUNT 3

float updateTime = 250;
volatile unsigned long stateChanges = 0;
volatile int lastState = 0;

unsigned long lastTime = 0;

int readHallState() {
  int a = digitalRead(HALL_A);
  int b = digitalRead(HALL_B);
  int c = digitalRead(HALL_C);
  return (a << 2) | (b << 1) | c;
}

void hallISR() {
  int s = readHallState();
  if (s != lastState) {
    stateChanges++;
    lastState = s;
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(HALL_A, INPUT_PULLUP);
  pinMode(HALL_B, INPUT_PULLUP);
  pinMode(HALL_C, INPUT_PULLUP);

  lastState = readHallState();

  attachInterrupt(digitalPinToInterrupt(HALL_A), hallISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(HALL_B), hallISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(HALL_C), hallISR, CHANGE);

  lastTime = millis();
}

void loop() {
  unsigned long now = millis();

  if (now - lastTime >= updateTime) {
    noInterrupts();
    unsigned long changes = stateChanges;
    stateChanges = 0;
    interrupts();

    float rpm = changes * (1000 / updateTime) / (POLE_COUNT * HALL_COUNT) * 60.0;

    // Serial.print("RPM: ");
    // Serial.println(rpm);
    Serial.print(rpm);
    Serial.print(",");
    Serial.println(lastState);

    lastTime = now;
  }
}