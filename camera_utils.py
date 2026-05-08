# camera_utils.py
import cv2

def list_available_cameras(max_test: int = 5):  # Reduced from 10
    """Scans and returns a list of available webcams - optimized for Windows."""
    print("🔍 Scanning for available webcams...")
    available = []
    
    for i in range(max_test):
        # CAP_DSHOW is much faster on Windows for Logitech and most webcams
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            cap.release()
            continue
            
        # Quick frame read to confirm camera is working
        ret, frame = cap.read()
        
        if ret and frame is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append((i, f"Camera {i} ({width}x{height})"))
        
        cap.release()
    
    return available


def select_camera():
    """Lists available cameras and lets user select one."""
    cameras = list_available_cameras()
    
    if not cameras:
        print("❌ No cameras found!")
        return None

    print("\n📷 Available Cameras:")
    for idx, (cam_id, name) in enumerate(cameras):
        print(f" {idx}: {name}")

    if len(cameras) == 1:
        selected = 0
        print(f"\n✅ Only one camera detected. Auto-selecting: {cameras[0][1]}")
    else:
        while True:
            try:
                choice = int(input(f"\nSelect camera (0 to {len(cameras)-1}): "))
                if 0 <= choice < len(cameras):
                    selected = choice
                    break
                else:
                    print("Invalid number. Try again.")
            except ValueError:
                print("Please enter a valid number.")

    cam_index = cameras[selected][0]
    print(f"✅ Selected Camera Index: {cam_index}\n")
    return cam_index