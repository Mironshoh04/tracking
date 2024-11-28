import cv2

def main():
    # Initialize video capture (0 for webcam, or provide video file path)
    # video_path = 'istockphoto-1248544042-640_adpp_is.mp4'
    video_path = 'istockphoto-1187482501-640_adpp_is.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Select ROI (Region of Interest) manually
    roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")

    # Extract the template (ROI)
    x, y, w, h = [int(v) for v in roi]
    template = frame[y:y+h, x:x+w]

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Perform template matching
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Draw a rectangle around the matched area
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Object Tracking", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
