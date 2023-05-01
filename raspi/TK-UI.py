import tkinter as tk 
import cv2,sys
from PIL import Image,ImageTk
import numpy as np
sys.path.append(r"E:\文档\目标检测工程集\garbage-classification-keras/") 
from detect import tflite_model

image_shape=(624,624,3)

def take_snapshot():
    pass

def video_loop():
    success, frame = cap.read()  # 从摄像头读取照片
    if success:
        # cv2.waitKey(0)
        # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#转换颜色从BGR到RGB
        # 进行检测
        out_boxes, out_scores, out_classes=my_yolo.detect_image(frame)
        frame = my_yolo.plotBox(out_boxes, out_scores, out_classes)

        imgtk = ImageTk.PhotoImage(image=frame)
        panel.config(image=imgtk)
        panel.image=imgtk
        root.after(100, video_loop)

cap = cv2.VideoCapture(0)    #摄像头,cv2.CAP_DSHOW
ret = cap.set(3, image_shape[0])  # 设置帧宽
ret = cap.set(4, image_shape[1])  # 设置帧高

my_yolo = tflite_model()

root = tk.Tk()
root.title("垃圾分类垃圾箱")

panel = tk.Label(root)  # initialize image panel
panel.grid(row=0, column=0,columnspan=7,rowspan=7,sticky='W')

root.config(cursor="arrow")
btn = tk.Button(root, text="点击开始", command=take_snapshot)
btn.grid(row=8, column=4,columnspan=3, sticky='W')

video_loop()
# 设置文本框高度为1，宽度为10
height_width_label = tk.Label(root, text='垃圾名称：')
height_width_text = tk.Text(root, height=1, width=10)
height_width_text.insert('0.0', '矿泉水瓶')
height_width_label.grid(row=0, column=8, sticky='E')
height_width_text.grid(row=0, column=9, sticky='W')

width_label = tk.Label(root, text='归属类别：')
width_text = tk.Text(root, height=1, width=10)
width_text.insert('0.0', '可回收垃圾')
width_label.grid(row=1, column=8, sticky='E')
width_text.grid(row=1, column=9, sticky='W')

root.mainloop()
# 当一切都完成后，关闭摄像头并释放所占资源
cap.release()
cv2.destroyAllWindows()
