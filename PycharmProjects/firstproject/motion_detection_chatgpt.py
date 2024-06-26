import cv2

# Open the video file
video_path = r"C:\Users\sreekanth.maramreddy\Desktop\ds\module05-contours\Applications\intruder_2.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'output_video2.avi'
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Threshold the foreground mask
    _, thresh = cv2.threshold(fg_mask, 15, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the frame into the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Motion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
