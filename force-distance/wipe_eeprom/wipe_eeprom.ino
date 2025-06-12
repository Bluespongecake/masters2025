#include <EEPROM.h>

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < EEPROM.length(); i++) {
    EEPROM.write(i, 0xFF); // or use EEPROM.update(i, 0xFF) to reduce EEPROM wear
  }
  Serial.println("EEPROM wiped.");
}

void loop() {
  // Nothing to do here
}