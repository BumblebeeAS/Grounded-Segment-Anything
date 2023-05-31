import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from PIL import Image

import hashlib

# GroundingDINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

# SAM
from segment_anything import SamPredictor, build_sam


def load_model_hf(
    repo_id="ShilongLiu/GroundingDINO",
    filename="groundingdino_swinb_cogcoor.pth",
    ckpt_config_filename="GroundingDINO_SwinB.cfg.py",
    device="cpu",
):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def load_sam(device="cpu"):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        os.system("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(
        device=torch.device(device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def random_color():
    return np.concatenate([np.random.random(3), np.array([0.6])], axis=0)


def show_masks(masks, img, colors):
    annotated_frame_pil = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
    for i in range(masks.size(0)):
        mask = masks[i, :, :, :].cpu().numpy()
        h, w = mask.shape[-2:]
        color = colors[i]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        annotated_frame_pil = Image.alpha_composite(
            annotated_frame_pil,
            Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA"),
        )
    return np.array(annotated_frame_pil)


def to_yolo(
    save_label,
    boxes_labels_masks,
    detection_path,
    segmentation_path,
    disable_segmentation=False,
    image_width=640,
    image_height=480,
):
    with open(detection_path, "w") as f, open(segmentation_path, "w") as f_seg:
        for cxcywh, label, mask in boxes_labels_masks:
            # label_id = label_to_label_id.get(label, None)
            # if label_id is None:
            #     label_id = max_label_id
            #     label_to_label_id[label] = label_id
            #     max_label_id += 1

            cx, cy, w, h = cxcywh
            f.write(f"{save_label} {cx} {cy} {w} {h}\n")
            if len(masks) > 0 and not disable_segmentation:
                # Use masks
                mask_arr = mask.cpu().numpy()[0, :, :].astype(np.uint8)
                contours, hierarchy = cv2.findContours(
                    mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                polygon = contours[0].reshape(-1, 2)

                if len(polygon) < 3:
                    continue

                # Normalize polygon
                widths = polygon[:, 0] / image_width
                heights = polygon[:, 1] / image_height
                points = np.stack([widths, heights], axis=1)

                # Convert to YOLO format
                label_str = f"{save_label} "
                for x, y in points:
                    label_str += f"{x} {y} "
                label_str = label_str.rstrip() + "\n"
                f_seg.write(label_str)    


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, default="imgs/2.jpg")
    parser.add_argument("text", type=str, default="red chevron")
    parser.add_argument("output_label", type=str, default="object")
    parser.add_argument("-b", "--box_thresh", type=float, default=0.5)
    parser.add_argument("-t", "--text_thresh", type=float, default=0.25)
    parser.add_argument("-o", "--output_path", type=str, default="outputs")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--disable_segmentation", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode. Save annotated images to output/",
    )

    args = parser.parse_args()

    device = "cuda" if not args.use_cpu else "cpu"

    BOX_TRESHOLD = args.box_thresh
    TEXT_TRESHOLD = args.text_thresh

    # Load models
    groundingdino_model = load_model_hf(device=device)
    if not args.disable_segmentation:
        sam_predictor = load_sam(device=device)

    # Get text prompt
    # prompt = " . ".join(args.text.split(" ") + [""]).lower().rstrip()
    prompt = args.text.lower().rstrip()
    print(f"Prompt: {prompt}")
    img_paths = []
    if os.path.isfile(args.input):
        img_paths.append(args.input)
    elif os.path.isdir(args.input):
        for f in os.listdir(args.input):
            img_paths.append(os.path.join(args.input, f))
    else:
        raise ValueError("Invalid input path")

    # Create output directory if needed
    if args.debug and os.path.isdir("output"):
        os.system("rm -rf output")

    if args.debug and not os.path.isdir("output"):
        os.mkdir("output")
    
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "detect"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "segment"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "images"), exist_ok=True)

    if args.debug:
        os.makedirs(os.path.join(args.output_path, "visualize"), exist_ok=True)
    
    


    max_label_id = 0
    label_to_label_id_mapping = {}

    colors = [random_color() for _ in range(10)]
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path)

        #md5sum * | awk '{print "mv", $2, $1 ".jpg"  }' | bash
        # run linux command
        name = img_name.split(".")[0]
        hash_name = hashlib.md5(open(img_path,'rb').read()).hexdigest()

        try:
            image_source, image = load_image(img_path)
        except:
            print(f"Failed to load {img_path}")
            continue
        os.system(f"cp {img_path} {os.path.join(args.output_path, 'images', hash_name + '.jpg')}")

        # Use DINO to generate boxes
        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )

        if boxes.shape[0] == 0:
            print(f"No boxes found for {img_name}")
            continue

        # Use SAM to generate masks
        if not args.disable_segmentation:
            sam_predictor.set_image(image_source)
            H, W, _ = image_source.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy, image_source.shape[:2]
            ).to(torch.device(device))
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        annotation_file_name = hash_name + ".txt"

        detection_path = os.path.join(
            args.output_path, "detect", annotation_file_name
        )
        segmentation_path = os.path.join(
            args.output_path, "segment", annotation_file_name
        )

        if args.disable_segmentation:
            masks = []

        to_yolo(
            args.output_label,
            list(zip(boxes, phrases, masks)),
            detection_path,
            segmentation_path,
            args.disable_segmentation,
            image_width=image_source.shape[1],
            image_height=image_source.shape[0],
        )


        if not args.disable_segmentation and args.debug:
            annotated_frame = annotate(
                image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
            )
            annotated_frame = show_masks(masks, annotated_frame, colors)
            cv2.imwrite(os.path.join(args.output_path, "visualize", hash_name + '.jpg'), annotated_frame)