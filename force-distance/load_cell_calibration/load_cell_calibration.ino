#include <HX711_ADC.h>

// HX711 pins
const int HX711_dout = 2;
const int HX711_sck = 3;

HX711_ADC LoadCell(HX711_dout, HX711_sck);

// Calibration variables
long tare_raw = 0;
float slope = 1.0;

// Averaging parameters
const int NUM_SAMPLES = 20;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Load Cell Calibration Tool");

  LoadCell.begin();
  LoadCell.start(2000);
  LoadCell.setSamplesInUse(16);

  // Tare measurement
  Serial.println("\n=== TARE PHASE ===");
  Serial.println("Ensure the scale is EMPTY.");
  Serial.println("Press any key to continue...");
  waitForUser();

  tare_raw = averageReading();
  Serial.print("Tare value (raw): ");
  Serial.println(tare_raw);

  // Weight measurement
  Serial.println("\n=== CALIBRATION PHASE ===");
  Serial.println("Place a known weight on the scale.");
  Serial.println("Press any key once the weight is stable...");
  waitForUser();

  long raw_with_weight = averageReading();
  long raw_delta = raw_with_weight - tare_raw;

  Serial.print("Measured raw delta: ");
  Serial.println(raw_delta);

  float known_weight = getKnownWeight();

  slope = known_weight / raw_delta;

  Serial.println("\n=== CALIBRATION COMPLETE ===");
  Serial.print("Calibration Slope: ");
  Serial.println(slope, 8);
  Serial.println("You can now use this slope for accurate weight readings.");
}

void loop() {
  // Nothing in loop; all calibration is done in setup.
}

long averageReading() {
  long total = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    LoadCell.update();
    total += LoadCell.getData();
    delay(50);
  }
  return total / NUM_SAMPLES;
}

void waitForUser() {
  while (!Serial.available()) {
    delay(10);
  }
  while (Serial.available()) Serial.read();  // Clear buffer
}

float getKnownWeight() {
  Serial.println("\nEnter the known weight in grams and press Enter:");
  while (!Serial.available()) {
    delay(10);
  }

  String input = Serial.readStringUntil('\n');
  input.trim();
  float weight = input.toFloat();

  Serial.print("You entered: ");
  Serial.print(weight);
  Serial.println(" grams");

  return weight;
}