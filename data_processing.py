import cv2
import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse
from datasets import Dataset
from torchvision.transforms import PILToTensor
import torch

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--start_second",type=float,default=0.0)
parser.add_argument("--end_second",type=float,default=30.0)
parser.add_argument("--frame_interval",type=int,default=100)
parser.add_argument("--path",type=str,default="/scratch/jlb638/spider/intospiderverse.mp4")
parser.add_argument("--dataset_name",type=str,default="jlbaker361/spider-test")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large",cache_dir=cache_dir)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",cache_dir=cache_dir)

IMAGE_STR="image"
TEXT_STR="text"
FRAME_STR="frame"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
# Function to extract frames 
def FrameCapture(args): 
    src_dict={
    IMAGE_STR:[],
    TEXT_STR:[],
    FRAME_STR:[]
    }
    # Path to video file 
    vidObj = cv2.VideoCapture(args.path) 

    length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = vidObj.get(cv2.CAP_PROP_FPS)

    print(f"length {length}")
    print(f"width {width}")
    print(f"height {height}")
    print(f"fps {fps}")
    print(f"seconds {length/fps}")

    # Used as counter variable 
    count = 0
    # checks whether frames were extracted 
    success = 1
  
    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()
        # Saves the frames with frame-count 
        #cv2.imwrite("frame%d.jpg" % count, image) 
  
        count += 1
        second=count/fps
        '''print(f"count {count}")
        print(f"second {second}")
        print(f" second > start_second {second > args.start_second}")
        print(f"count % frame_interval {count % args.frame_interval}")
        print(f"second > end_second {second > args.end_second}")'''
        if second > args.start_second and count %args.frame_interval==0:
            color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            tensor_image=PILToTensor()(pil_image)
            tensor_image=tensor_image.to(device)
            inputs = processor(pil_image, return_tensors="pt") #.to(device)

            out = model.generate(**inputs)
            caption=processor.decode(out[0], skip_special_tokens=True)
            print(f"count: {count}")
            src_dict[IMAGE_STR].append(pil_image)
            src_dict[TEXT_STR].append(caption)
            src_dict[FRAME_STR].append(count)
        if second > args.end_second:
            return Dataset.from_dict(src_dict).push_to_hub(args.dataset_name)
    return Dataset.from_dict(src_dict).push_to_hub(args.dataset_name)

if __name__ == '__main__':
    # Calling the function 
    args = parser.parse_args()
    print(args)
    FrameCapture(args)