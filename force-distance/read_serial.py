import serial

# Replace this with your actual port
port = '/dev/cu.usbmodem2101'  # macOS port from your screenshot
baud = 115200

try:
    with serial.Serial(port, baud, timeout=1) as ser:
        print(f"Connected to {port} at {baud} baud.")
        print("Reading serial data. Press Ctrl+C to stop.\n")
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)

except serial.SerialException as e:
    print(f"Serial error: {e}")
except KeyboardInterrupt:
    print("\nSerial reading stopped.")