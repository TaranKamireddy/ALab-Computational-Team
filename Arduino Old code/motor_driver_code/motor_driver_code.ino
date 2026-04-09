// // pin connections
const int dirPin_zaxis = 3; // direction pin for zaxis
const int stepPin_zaxis = 4; // step pin for zaxis

const int dirPin_sy = 6; // direction pin for scotch yoke
const int stepPin_sy = 7; // step pin for scoth yoke
const int steps_z = 1250; //Steps for the z axis
const int steps_s = 50; //Steps for the scotch yoke 


void setup() {
  pinMode(dirPin_zaxis, OUTPUT);
  pinMode(stepPin_zaxis, OUTPUT);
  pinMode(dirPin_sy, OUTPUT);
  pinMode(stepPin_sy, OUTPUT);
  // set direction of rotation to clockwise
  digitalWrite(dirPin_sy, LOW); 
  digitalWrite(dirPin_zaxis, LOW);  
 
}



void loop() {
  delay(8000);
  digitalWrite(dirPin_zaxis, LOW); //Sets the direction of zaxis to go up
  for (int i = 0; i < steps_z; i++) {
    digitalWrite(stepPin_zaxis, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPin_zaxis, LOW);
    delayMicroseconds(1000);
  }
  delay(3000);
  //Sets the z axis to go down
  digitalWrite(dirPin_zaxis, HIGH);
  for (int i = 0; i < steps_z - 1; i++) {
    digitalWrite(stepPin_zaxis, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPin_zaxis, LOW);
    delayMicroseconds(1000); 
  }
  delay(1000);

  digitalWrite(dirPin_sy, HIGH);
  for (int i = 0; i < steps_s; i++) {
    digitalWrite(stepPin_sy, HIGH);
    delayMicroseconds(4000);
    digitalWrite(stepPin_sy, LOW);
    delayMicroseconds(4000);
  }
  delay(200);
  digitalWrite(dirPin_sy, LOW);
  for (int i = 0; i < steps_s; i++) {
    digitalWrite(stepPin_sy, HIGH);
    delayMicroseconds(4000);
    digitalWrite(stepPin_sy, LOW);
    delayMicroseconds(4000);
  }

  delay(1000); 
}
