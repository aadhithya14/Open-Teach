import cv2
import numpy as np

def test_camera_index(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame from camera at index {index}")
            break

        print( "Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
        cv2.imshow(f"Camera {index}", frame)
        print(np.asanyarray(frame).shape)

        if cv2.waitKey(1) == ord('q'):
            break
        # print(index)
        # break

    cap.release()
    cv2.destroyAllWindows()

# Test camera indices from 0 to 100
if __name__ == "__main__":
    for camera_id in range(100):
        test_camera_index(camera_id)