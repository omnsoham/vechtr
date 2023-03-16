
import pyttsx3
from gpiozero import Button
from gpiozero import Buzzer
from time import sleep

button = Button(11)
motor1 = Buzzer(12) 
motor2 = Buzzer(13)
motor3 = Buzzer(16)
text_speech = pyttsx3.init()
all = 0
bathroom = 1
chair = 2
door = 3
counter = 0
mode = counter%4



while True:
    if button.is_pressed:
        if (mode == all):
            text_speech.say("all")
            text_speech.runAndWait()
        elif(mode == bathroom):
            text_speech.say("bathroom")
            text_speech.runAndWait()               
        elif(mode == chair):
            text_speech.say("chair")
            text_speech.runAndWait()               
        elif(mode == door):
            text_speech.say("door")
            text_speech.runAndWait()               
        counter += 1
    else:
        print(".")
    sleep(1)
