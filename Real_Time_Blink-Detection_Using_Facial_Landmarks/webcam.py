from landmarks import EAR
from collections import deque
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    EARs = deque(maxlen=13)
    while True:
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            continue
        EARs.append(EAR(image))
        if(len(EARs) < 13):
            print("Deque still being filled")
        image = cv2.flip(image,1)
        cv2.imshow("Output", image)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.release()