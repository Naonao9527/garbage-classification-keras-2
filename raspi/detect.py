import numpy as np
import cv2,sys,time
from tensorflow import lite
from PIL import Image

sys.path.append(r"E:\文档\目标检测工程集\garbage-classification-keras/") 
from utils_bbox import DecodeBox
import config

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

class tflite_model():
    def __init__(self) -> None:
        self.interpreter = lite.Interpreter(model_path=config.litemodel_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
 
    def detect_image(self,image):
        self.r_image=image.copy()       #留下原图像，以便后续展示图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # self.r_image=image.copy()       #留下原图像，以便后续展示图片
        image=Image.fromarray(image)

        image=resize_image(image,config.input_shape,True)
        image=np.array(image, dtype='float32')
        #归一化
        image = image / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)


        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        out_put1 = self.interpreter.get_tensor(self.output_details[0]['index'])
        out_put2 = self.interpreter.get_tensor(self.output_details[1]['index'])
        out_put3 = self.interpreter.get_tensor(self.output_details[2]['index'])
        out_puts = [out_put2,out_put1,out_put3]
        self.out_boxes, self.out_scores, self.out_classes=DecodeBox(out_puts,
                                                                    config.anchors,
                                                                    config.num_classes,
                                                                    config.input_shape,
                                                                    self.r_image.shape[:2])

    def video_show(self,creame_id): 
        capture = cv2.VideoCapture(creame_id)    #摄像头,cv2.CAP_DSHOW
        ret = capture.set(3, 624)  # 设置帧宽
        ret = capture.set(4, 624)  # 设置帧高
        fps = 0.0
        while True:
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 进行检测
            self.detect_image(frame)
            frame=self.plotBox()
            # RGBtoBGR满足opencv显示格式

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
    
    def plotBox(self): 
    #---------------------------------------------------------#
    #   图像绘制
    #---------------------------------------------------------#
        image=self.r_image
        
        for i, c in list(enumerate(self.out_classes)):
            predicted_class = config.class_names[int(c)]
            box             = self.out_boxes[i]
            score           = self.out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.shape[1], np.floor(bottom).astype('int32'))
            right   = min(image.shape[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            # label = label.encode('utf-8')
            print(label, top, left, bottom, right)
        
            # cv2.putText(image, label, box[:2], cv2.FONT_HERSHEY_PLAIN,
            #     3, (0,0,255), 3)
            cv2.rectangle(image,(left,top),( right,bottom),(0,0,255),2)
        return image
    
if __name__ =="__main__":

    mymodel=tflite_model()
    mymodel.video_show(0)

    # image = cv2.imread('img/1.jpg')
    # mymodel=tflite_model()
    # mymodel.detect_image(image)
    # r_image=mymodel.plotBox()
    # cv2.imshow("image",r_image)
    # cv2.waitKey(0)
