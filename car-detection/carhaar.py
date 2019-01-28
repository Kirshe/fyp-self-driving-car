import cv2

# cap = cv2.VideoCapture('challenge.mp4') 
frames = cv2.imread('toycar2.jpg')
car_cascade = cv2.CascadeClassifier('cars.xml') 

while True: 
	 
	# ret, frames = cap.read() 
	# if not ret:
		# break
	# frames = cv2.resize(frames, (600,400))
	gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
	
	dist = []
	cars = car_cascade.detectMultiScale(gray, 1.1, 1, minSize = (50, 50))
	
	for (x,y,w,h) in cars: 
		cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
		dist.append([x+w//4, y+h//2, w*h])

		cv2.putText(frames, f"{dist[-1][2]}", (dist[-1][0], dist[-1][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), thickness = 2)


	cv2.imshow('video2', frames) 
	if cv2.waitKey(33) == ord('q'): 
		break

cv2.destroyAllWindows() 

