#include<Servo.h>
#include<LiquidCrystal_I2C.h>

#define ForwardMax 1910
#define ReverseMin 1048
#define NeutralMid 1479
#define MidTest01 1500

LiquidCrystal_I2C lcd(0x27, 16, 2); // I2C address 0x27, 16 column and 2 rows
Servo esc; // create servo class with the name as "esc"

void setup() {

int count = 0;

lcd.init(); // initialize the lcd
lcd.backlight(); // turn on back light

esc.attach(9); // Specify the esc signal pin, here as 9

}

void loop() {
lcd.print("Reading code");
esc.writeMicroseconds(NeutralMid); // set neutral
delay(4500);

lcd.clear();

lcd.print("Initiating Dwell");
//delay(3500);

lcd.setCursor(0, 1);
int count = (50000) / 1000;
for (int i = count; i >= 0; i--) {

if (count < 10)
lcd.print(' '); // alternatively: '0'
lcd.clear();
lcd.print(i); 
lcd.print(" Seconds");
delay(1000);
}

lcd.clear();
lcd.print("Dwell Complete");
delay(1000);

lcd.clear();
lcd.setCursor(0, 0);

lcd.print("Ramping up");
delay(1050);
lcd.clear();
//esc.writeMicroseconds(NeutralMid*1.005); // Forward throttle
//delay(1050);
esc.writeMicroseconds(NeutralMid*1.01); // Forward throttle
delay(2050);
lcd.print("5% throttle");
lcd.setCursor(1, 3);
lcd.print("RPM = 5000");
delay(2050);

lcd.setCursor(0, 0);



esc.writeMicroseconds(NeutralMid*1.05); // Forward throttle // 5% throttle is minimum registering throttle for command line to ESC.
delay(10000);
lcd.clear();

lcd.print("end cycle");
delay(1000);
lcd.clear();


lcd.print("Refresh cycle");
delay(5000); // Code refresh cycle duration in milliseconds
lcd.clear();
delay(1000);

}