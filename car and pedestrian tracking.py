import cv2

#get video footage
video = cv2.VideoCapture('tesla dashboard2.mp4')
#video = cv2.VideoCapture('pedestrains.mp4')

#load some pre_trained data on car rear ends (haar cascade algorithm)
car_tracker = cv2.CascadeClassifier('car detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade fullbody.xml')

#Run forever until car stops or something
while True:
	
    #read the current frame
    (read_successful, frame) = video.read() 

    if read_successful:
        #must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    #detect pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    
    #draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (225, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 2)
        
    #draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 225), 2)

    #display the image with the faces spotted
    cv2.imshow('car and pedestrain detector', frame)

    #dont autoclose (wait here in the code and listen for  a key press)
    cv2.waitKey(1)
            
"""
#create opencv image
img = cv2.imread(img_file)

#create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#covert ot grayscale(needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)

#Draw rectangles aroud the cars
for (x, y, w, h) in cars:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 225, 0), 2)

#display the image with the faces spotted
cv2.imshow('car detector', img)

#dont autoclose (wait here in the code and listen for  a key press)
cv2.waitKey()
"""

print("code completed")