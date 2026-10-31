#!/usr/bin/env python

"""
Direct Raw Serial Diagnostic for Feetech Servos (COM8 Follower).
Tests all baud rates and all motor IDs using raw serial bytes (no dependencies).
"""

import sys
import time
import serial


def calculate_ping_packet(motor_id: int) -> bytes:
    # Feetech Protocol 0 PING packet:
    # [0xFF, 0xFF, ID, Length=2, Instruction=1, Checksum]
    # Checksum = ~(ID + Length + Instruction) & 0xFF
    checksum = (~(motor_id + 2 + 1)) & 0xFF
    return bytes([0xFF, 0xFF, motor_id, 0x02, 0x01, checksum])


def scan_port(port: str = "COM8"):
    baudrates = [
        1_000_000,
        500_000,
        250_000,
        128_000,
        115_200,
        57_600,
        38_400,
        19_200,
        9_600,
    ]

    print("=" * 60)
    print(f" Scanning Follower Arm Port: {port}")
    print("=" * 60)

    found_any = False

    for baud in baudrates:
        sys.stdout.write(f"Testing {baud:>9,} baud... ")
        sys.stdout.flush()

        try:
            ser = serial.Serial(port, baudrate=baud, timeout=0.08)
        except Exception as e:
            print(f"Cannot open port ({e})")
            continue

        responded_ids = []

        for motor_id in range(1, 7):
            ser.reset_input_buffer()
            pkt = calculate_ping_packet(motor_id)
            ser.write(pkt)
            time.sleep(0.015)

            # Read response
            resp = ser.read(ser.in_waiting or 6)
            if resp and len(resp) >= 6 and resp[0] == 0xFF and resp[1] == 0xFF:
                # resp[2] is the motor ID returning status
                responded_ids.append(resp[2])

        ser.close()

        if responded_ids:
            print(f"--> [SUCCESS] Responding Motor IDs: {responded_ids}")
            found_any = True
        else:
            print("No response")

    print("=" * 60)
    if found_any:
        print("[✓] Motors detected! Note the baudrate above.")
    else:
        print("[!] No motors responded on any baudrate.")
        print("\nPossible hardware issues to check:")
        print(" 1. Board Switch: Is there a USB/UART or Power switch on the board that is OFF?")
        print(" 2. 3-Pin Cable: Is the 3-pin cable (GND, 12V, Data) plugged into the correct SERVO port?")
        print(" 3. Adapter Type: Does the replacement chip have an automatic half-duplex transceiver?")
        print(" 4. Latency Timer: Set Windows Latency Timer to 1ms in Device Manager.")
    print("=" * 60)


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "COM8"
    scan_port(port)
