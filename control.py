"""
For more information on the serial module, see: https://pythonhosted.org/pyserial/index.html
"""

import time

import serial


def main() -> None:
    """
    Initialize the serial connection
    """
    global uart
    uart = serial.Serial(
        port="/dev/ttyS0",
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        writeTimeout=1.0,
    )


def forward():
    uart.write(b"w")


def left():
    uart.write(b"a")


def backward():
    uart.write(b"s")


def right():
    uart.write(b"d")


def stop():
    uart.write(b"o")
    time.sleep(0.1)
    uart.write(b"o")
