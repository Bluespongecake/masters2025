#include <AccelStepper.h>

// Stepper motor pins
const int PUL = 7;  // Pulse pin
const int DIR = 6;  // Direction pin
const int ENA = 5;  // Enable pin

// Create stepper object using DRIVER mode
AccelStepper stepper(AccelStepper::DRIVER, PUL, DIR);

// Speed settings
const float MAX_SPEED = 5000.0;  // steps per second
const float MOVE_SPEED = 1000.0;  // steps per second

void setup() {
  Serial.begin(115200);
  pinMode(ENA, OUTPUT);
  digitalWrite(ENA, HIGH);  // Enable the motor driver

  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(1000);  // Optional: add acceleration
  stepper.setSpeed(0);

  Serial.println("Stepper Ready. Use 'w' to move up, 's' to move down.");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == 'w' || cmd == 'W') {
      stepper.setSpeed(-MOVE_SPEED);  // Move up
      Serial.println("Moving up...");
    } 
    else if (cmd == 's' || cmd == 'S') {
      stepper.setSpeed(MOVE_SPEED);   // Move down
      Serial.println("Moving down...");
    } 
    else if (cmd == 'x' || cmd == 'X') {
      stepper.setSpeed(0);            // Stop motor
      Serial.println("Motor stopped.");
    }
  }

  // Continually run the motor at the set speed
  stepper.runSpeed();
}