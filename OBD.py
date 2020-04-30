#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Thu Apr 30 21:32:04 2020

@author: Mohammed Shukri
"""

import obd

connection = obd.OBD() # auto-connects to USB 

cmd = obd.commands.SPEED # select an OBD command (sensor) in this case we choose the speed command to display current car's speed.


#ports = obd.scan_serial()       # return list of valid USB or RF ports
#print (ports)                    # ['/dev/ttyUSB0', '/dev/ttyUSB1']

# a while loop looping forever to continuosly print the car's speed.
while(1):
    response = connection.query(cmd) # send the command, and parse the response
    
    print(response.value) # returns unit-bearing values 