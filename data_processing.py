import cv2
import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration,Pix2StructForConditionalGeneration, Pix2StructProcessor
import argparse
from datasets import Dataset
from torchvision.transforms import PILToTensor
from torchvision import transforms
import torch

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--start_second",type=float,default=0.0)
parser.add_argument("--end_second",type=float,default=30.0)
parser.add_argument("--frame_interval",type=int,default=100)
parser.add_argument("--path",type=str,default="/scratch/jlb638/spider/")
parser.add_argument("--dataset_name",type=str,default="jlbaker361/spider-test")
parser.add_argument("--center_crop",default=False,action="store_true",)
parser.add_argument("--resolution",type=int,default=512)
parser.add_argument("--caption_model",type=str,default="blip")

IMAGE_STR="image"
TEXT_STR="text"
FRAME_STR="frame"
TITLE_STR="title"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
# Function to extract frames 
def FrameCapture(args): 
    src_dict={
    IMAGE_STR:[],
    TEXT_STR:[],
    FRAME_STR:[],
    TITLE_STR:[]
    }
    if args.caption_model=='blip':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large",cache_dir=cache_dir)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",cache_dir=cache_dir)
    elif args.caption_model=="pix2":
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base",cache_dir=cache_dir)
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base",cache_dir=cache_dir)
    # Path to video file
    for mp4_file in os.listdir(args.path):
        if mp4_file.find("mp4")==-1:
            continue
        vidObj = cv2.VideoCapture(os.path.join(args.path, mp4_file)) 

        length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = vidObj.get(cv2.CAP_PROP_FPS)

        print(f"file name {mp4_file}")
        print(f"length {length}")
        print(f"width {width}")
        print(f"height {height}")
        print(f"fps {fps}")
        print(f"seconds {length/fps}")

        train_transforms = transforms.Compose(
                [
                    transforms.Resize(int(args.resolution*1), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                ]
            )


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
                pil_image=train_transforms(pil_image)
                '''tensor_image=PILToTensor()(pil_image)
                tensor_image=tensor_image.to(device)'''
                inputs = processor(pil_image, return_tensors="pt") #.to(device)

                out = model.generate(**inputs)
                caption=processor.decode(out[0], skip_special_tokens=True)
                print(f"count: {count}")
                if caption.find("batman")==-1:
                    src_dict[IMAGE_STR].append(pil_image)
                    src_dict[TEXT_STR].append(caption.replace("anime","").replace("cartoon",""))
                    src_dict[FRAME_STR].append(count)
                    src_dict[TITLE_STR].append(mp4_file)
            if second > args.end_second:
                break
    return Dataset.from_dict(src_dict).push_to_hub(args.dataset_name)

if __name__ == '__main__':
    # Calling the function 
    args = parser.parse_args()
    print(args)
    FrameCapture(args)