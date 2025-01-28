import cv2

# Load Haar Cascade file
a = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
b = cv2.VideoCapture(0)
if not b.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    c_rec, d_image = b.read()
    if not c_rec:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    f = a.detectMultiScale(e, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d_image, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)

    # Show the video feed
    cv2.imshow('Face Detection', d_image)

    # Break loop if 'ESC' key (27) is pressed
    h = cv2.waitKey(1) & 0xFF
    if h == 27:
        break

# Release resources
b.release()
cv2.destroyAllWindows()
