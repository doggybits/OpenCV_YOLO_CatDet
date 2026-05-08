# cat_color_debug.py
import cv2
import numpy as np


class CatColorDebugger:
    """Separate module for debugging cat color classification"""
    
    window_name = "Cat Color Debug"
    enabled = False
    last_debug_time = 0

    @staticmethod
    def enable():
        CatColorDebugger.enabled = True
        print("🐱 Cat Color Debug Mode ENABLED")

    @staticmethod
    def disable():
        CatColorDebugger.enabled = False
        cv2.destroyWindow(CatColorDebugger.window_name)
        print("🐱 Cat Color Debug Mode DISABLED")

    @staticmethod
    def toggle():
        if CatColorDebugger.enabled:
            CatColorDebugger.disable()
        else:
            CatColorDebugger.enable()

    @staticmethod
    def show_debug_info(center_crop, orange_mask, white_mask,
                       orange_ratio, white_ratio, brightness,
                       label, confidence):
        
        if not CatColorDebugger.enabled:
            return

        # Create debug visualization
        debug_vis = np.zeros((400, 800, 3), dtype=np.uint8)

        # Original center crop
        small_vis = cv2.resize(center_crop, (200, 200))
        debug_vis[0:200, 0:200] = small_vis

        # Orange mask
        orange_vis = cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR)
        orange_vis = cv2.resize(orange_vis, (200, 200))
        debug_vis[0:200, 200:400] = orange_vis
        cv2.putText(debug_vis, "Orange Mask", (210, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # White mask
        white_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        white_vis = cv2.resize(white_vis, (200, 200))
        debug_vis[0:200, 400:600] = white_vis
        cv2.putText(debug_vis, "White Mask", (410, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Info panel
        cv2.putText(debug_vis, f"Label: {label}", (10, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_vis, f"Confidence: {confidence:.3f}", (10, 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(debug_vis, f"Orange Ratio: {orange_ratio:.3f}", (10, 310),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(debug_vis, f"White Ratio : {white_ratio:.3f}", (10, 340),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_vis, f"Brightness  : {brightness:.1f}", (10, 370),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow(CatColorDebugger.window_name, debug_vis)


# Optional: Auto enable when imported (uncomment if you want)
# CatColorDebugger.enable()