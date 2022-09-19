import cv2
import numpy as np
import smtplib
import playsound
import threading

statusAlarm = False
statusEmail = False
fire = 0

def play_alarm_sound_function(): #This function will play the alarm sound in case of fire/smoke
    while True:
        playsound.playsound('alarm-sound.mp3',True)

#This particular function is not working for the time being...
#This function will send a mail as an SOS in case of emergency
# def send_mail_function():

#     recipientEmail = "Reciver@gmail.com"
#     recipientEmail = recipientEmail.lower()

#     try:
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.ehlo()
#         server.starttls()
#         server.login("Sender@gmail.com", "Sender_Password")
#         server.sendmail('Sender@gmail.com', recipientEmail, "Warning A Fire Accident has been reported on ABC Company")
#         print("sent to {}".format(recipientEmail))
#         server.close()
#     except Exception as e:
#         print(e)


#Since detecting fire at home is not possible we have given a video as an input.
#We can give 0 or 1 as input to use webcam 
video = cv2.VideoCapture('smokeVid.mp4') #Replace smokeVid.mp4 with 0 to use webcam
#video = cv2.VideoCapture('false.mp4') # To test the product on false cases

while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    frame = cv2.resize(frame, (960, 540))

    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(frame, hsv, mask=mask)

    no_red = cv2.countNonZero(mask)

    if int(no_red) > 15000:
        fire = fire + 1

    cv2.imshow("output", output)

    if fire >= 1:

        if statusAlarm == False:
            threading.Thread(target=play_alarm_sound_function).start()
            statusAlarm = True

        # if statusEmail == False:
        #     threading.Thread(target=send_mail_function).start()
        #     statusEmail = True


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()