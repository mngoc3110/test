import argparse
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F

from models.Generate_Model import GenerateModel
from models.Text import *
from dataloader.video_dataloader import test_data_loader
from utils.builders import get_class_info

def build_model(args, input_text):
    model = GenerateModel(input_text=input_text, args=args)
    return model

def validate_with_logit_adjustment(val_loader, model, device, class_counts, tau=1.0):
    model.eval()
    all_preds = []
    all_targets = []
    
    # Calculate Logit Adjustment
    # adjustment = log(p_y) * tau
    # We want to subtract this from the logits to penalize frequent classes
    class_counts = torch.tensor(class_counts, device=device, dtype=torch.float32)
    prior_prob = class_counts / class_counts.sum()
    logit_adjustment = torch.log(prior_prob + 1e-8) * tau
    
    print(f"\nEvaluating with Logit Adjustment (tau={tau})...")
    print(f"Prior Probs: {prior_prob.cpu().numpy()}")
    print(f"Adjustment: {logit_adjustment.cpu().numpy()}")

    with torch.no_grad():
        for i, (images_face, images_body, target) in enumerate(tqdm(val_loader)):
            images_face = images_face.to(device)
            images_body = images_body.to(device)
            target = target.to(device)

            output, _, _, _ = model(images_face, images_body)
            
            # Apply adjustment
            # Logit Adjustment: logit = logit - log(p_y) * tau
            # This encourages the model to predict rare classes more often
            adjusted_output = output - logit_adjustment
            
            preds = adjusted_output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    cm = confusion_matrix(all_targets, all_preds)
    acc = np.trace(cm) / np.sum(cm) * 100
    class_acc = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)
    uar = np.nanmean(class_acc) * 100
    
    return acc, uar, cm

def main():
    parser = argparse.ArgumentParser(description='Evaluate with Logit Adjustment')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='CAER', help='Dataset name')
    parser.add_argument('--root-dir', type=str, default='./dataset/CAER', help='Root directory')
    parser.add_argument('--test-annotation', type=str, default='./dataset/CAER/annotations/test.txt', help='Path to test annotation')
    parser.add_argument('--train-annotation', type=str, default='./dataset/CAER/annotations/train.txt', help='Path to train annotation (for class counts)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    # Model args needed for building
    parser.add_argument('--text-type', default='prompt_ensemble')
    parser.add_argument('--temporal-pooling', default='attn_pool')
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--contexts-number', type=int, default=8)
    parser.add_argument('--class-token-position', type=str, default='end')
    parser.add_argument('--class-specific-contexts', type=str, default='True')
    parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True')
    parser.add_argument('--num-segments', type=int, default=4)
    parser.add_argument('--duration', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--clip-path', type=str, default='ViT-B/16')
    parser.add_argument('--bounding-box-face', type=str, default='./dataset/CAER/bounding_box/face.json')
    parser.add_argument('--bounding-box-body', type=str, default='./dataset/CAER/bounding_box/body.json')
    parser.add_argument('--crop-body', action='store_true')
    parser.add_argument('--use-moco', action='store_true')
    parser.add_argument('--moco-k', type=int, default=4096)
    parser.add_argument('--moco-m', type=float, default=0.99)
    parser.add_argument('--moco-t', type=float, default=0.07)

    args = parser.parse_args()
    
    # Setup Device
    if args.gpu == 'mps':
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    args.device = device

    # 1. Get Class Info
    class_names, input_text = get_class_info(args)
    num_classes = len(class_names)
    
    # 2. Count Class Frequencies from Train Set (Needed for Logit Adjustment)
    print(f"=> Counting class frequencies from {args.train_annotation}...")
    class_counts = [0] * num_classes
    # Try reading the file directly to be faster than loading dataset
    try:
        with open(args.train_annotation, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = int(parts[-1])
                    # Handle 1-based if necessary (simple heuristic)
                    if args.dataset == 'DAiSEE' and label >= 4: label = 3 # Clamp fix
                    if label < num_classes:
                        class_counts[label] += 1
    except Exception as e:
        print(f"Error reading train file: {e}. Using uniform distribution (no adjustment).")
        class_counts = [1] * num_classes

    # Handle zero counts
    class_counts = [c if c > 0 else 1 for c in class_counts]
    print(f"Class Counts: {class_counts}")

    # 3. Build Model & Load Checkpoint
    model = build_model(args, input_text)
    model = model.to(device)
    
    print(f"=> Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    # Clean state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    # 4. Load Test Data
    test_data = test_data_loader(
        root_dir=args.root_dir,
        list_file=args.test_annotation,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 5. Evaluate with different Tau values
    taus = [0.0, 0.5, 1.0, 1.2, 1.5, 2.0]
    results = []
    
    print("\n" + "="*50)
    print(f"STARTING LOGIT ADJUSTMENT EVALUATION on {args.dataset}")
    print("="*50)

    for tau in taus:
        war, uar, cm = validate_with_logit_adjustment(test_loader, model, device, class_counts, tau=tau)
        print(f"\n>>> Tau: {tau} | WAR: {war:.2f}% | UAR: {uar:.2f}%")
        print(f"Confusion Matrix:\n{cm}")
        results.append((tau, war, uar))

    # 6. Summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"{'Tau':<10} | {'WAR (Acc)':<15} | {'UAR':<15}")
    print("-" * 45)
    best_war = (0, 0)
    best_uar = (0, 0)
    
    for tau, war, uar in results:
        print(f"{tau:<10} | {war:<15.2f} | {uar:<15.2f}")
        if war > best_war[1]: best_war = (tau, war)
        if uar > best_uar[1]: best_uar = (tau, uar)
    
    print("-" * 45)
    print(f"Best WAR: {best_war[1]:.2f}% at Tau={best_war[0]}")
    print(f"Best UAR: {best_uar[1]:.2f}% at Tau={best_uar[0]}")

if __name__ == '__main__':
    main()
