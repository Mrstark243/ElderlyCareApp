import cv2
import os

def test_face_detection():
    try:
        # Check for haarcascades
        if hasattr(cv2, 'data'):
            print(f"CV2 Data Path: {cv2.data.haarcascades}")
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            if os.path.exists(cascade_path):
                print("Haar Cascade found.")
                face_cascade = cv2.CascadeClassifier(cascade_path)
                print("Classifier loaded successfully.")
            else:
                print("Haar Cascade XML not found.")
        else:
            print("CV2 data attribute missing.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_face_detection()
