import cv2

# Provide the correct path to the Haar cascade XML file
alg = "C:/Users/HP/Downloads/haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Initialize camera
cam = cv2.VideoCapture(0)  # Use 0 instead of 1 unless you have multiple cameras

while True:
    _, img = cam.read()  # Read the frame from the camera
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert color to grayscale

    # Detect faces in the image
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with rectangles
    cv2.imshow("FaceDetection", img)

    # Check for the 'Esc' key to exit the loop
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the camera and close the OpenCV windows
cam.release()
cv2.destroyAllWindows()