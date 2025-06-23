import tkinter as tk

from PIL import ImageGrab
from inferencing import inferScreenShot


class WhiteboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whiteboard")

        self.canvas = tk.Canvas(root, width=780, height=780, bg="white")
        self.canvas.pack()

        self.draw_button = tk.Button(root, text="Draw", command=self.start_draw)
        self.draw_button.pack(side=tk.LEFT)

        self.erase_button = tk.Button(root, text="Erase", command=self.start_erase)
        self.erase_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.info_button = tk.Button(root, text="Info", command=self.print_canvas_info)
        self.info_button.pack(side=tk.LEFT)

        self.screenshot_button = tk.Button(root, text="ScreenShot", command=self.screenShot)
        self.screenshot_button.pack(side=tk.LEFT)

        self.drawing = False
        self.erasing = False
        self.last_x, self.last_y = None, None

        self.canvas.bind("<Button-1>", self.start_action)
        self.canvas.bind("<B1-Motion>", self.draw_or_erase)
        self.canvas.bind("<ButtonRelease-1>", self.stop_action)

    def screenShot(self, filename="ScreenShot.png"):
        self.root.update()
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        image = ImageGrab.grab(bbox=(x, y, x1, y1))
        image.save(filename)

        inferScreenShot()

    def print_canvas_info(self):
        self.root.update_idletasks()  # Make sure geometry is up-to-date
        print("Canvas info:")
        print(f" - Window-relative position: ({self.canvas.winfo_x()}, {self.canvas.winfo_y()})")
        print(f" - Screen position: ({self.canvas.winfo_rootx()}, {self.canvas.winfo_rooty()})")
        print(f" - Size: {self.canvas.winfo_width()} x {self.canvas.winfo_height()}")

    def start_action(self, event):
        if self.drawing:
            self.start_draw(event)
        elif self.erasing:
            self.start_erase(event)

    def stop_action(self, event):
        if self.drawing:
            self.stop_draw(event)
        elif self.erasing:
            self.stop_erase(event)

    def start_draw(self, event=None):
        self.drawing = True
        self.erasing = False
        if event:
            self.last_x, self.last_y = event.x, event.y

    def stop_draw(self, event=None):
        self.drawing = False
        self.erasing = False

    def start_erase(self, event=None):
        self.erasing = True
        self.drawing = False
        if event:
            self.last_x, self.last_y = event.x, event.y

    def stop_erase(self, event=None):
        self.erasing = False
        self.drawing = False

    def draw_or_erase(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=20)
            self.last_x, self.last_y = x, y
        elif self.erasing:
            x, y = event.x, event.y
            self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")


if __name__ == "__main__":
    root = tk.Tk()
    app = WhiteboardApp(root)
    root.mainloop()
