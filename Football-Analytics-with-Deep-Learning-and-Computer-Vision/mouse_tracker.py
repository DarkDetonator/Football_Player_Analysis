import cv2
import matplotlib.pyplot as plt

def show_image_with_coords(img_path):
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return
    
    # Convert BGR (OpenCV) to RGB (Matplotlib)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize plot
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("Click on the image (0 clicks)")

    # List to store click coordinates
    clicks = []
    
    def on_click(event):
        if event.xdata and event.ydata:
            # Append clicked coordinates
            clicks.append((event.xdata, event.ydata))
            
            # Clear and redraw image
            ax.clear()
            ax.imshow(img)
            
            # Plot all clicked points
            for x, y in clicks:
                ax.plot(x, y, 'go', markersize=8)  # Green dot for each click
            
            # Update title with click count and latest coordinates
            latest_x, latest_y = int(event.xdata), int(event.ydata)
            plt.title(f"Clicked at ({latest_x}, {latest_y}) | Total clicks: {len(clicks)}")
            fig.canvas.draw_idle()
            
            # Print coordinates to console
            print(f"Click {len(clicks)}: ({latest_x}, {latest_y})")

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

# Replace with the correct path to your image
image_path = "Football_Player_Analysis/Football-Analytics-with-Deep-Learning-and-Computer-Vision/tactical map 2.jpg"
show_image_with_coords(image_path)