from landmarks import EAR
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            continue
        print(EAR(image))
        image = cv2.flip(image,1)
        cv2.imshow("Output", image)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.release()