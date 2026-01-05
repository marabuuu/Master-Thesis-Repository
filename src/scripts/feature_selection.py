import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/data/horse/ws/mala059b-rna2wsi/mopadi/src")
from mopadi.mil.utils import Classifier

def extract_patient_id(filename, index):
    return "-".join(filename.split("-")[:index])

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Select top features per patient using a trained MIL classifier.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    params = config['feature_selection']

    feats_dir = params['feats_dir']
    mil_weights = params['mil_weights']
    clini_table = params['clini_table']
    target_label = params['target_label']
    target_dict = params['target_dict']
    n_top_features = params['n_top_features']
    output_csv = params['output_csv']
    fname_index = params['fname_index']
    device = params.get('device', 'cuda:0')
    classifier_params = params.get('classifier_params', {})

    # Load clinical table
    clini_df = pd.read_csv(clini_table)
    clini_df = clini_df.dropna(subset=[target_label])
    clini_df = clini_df[clini_df[target_label].isin(target_dict.keys())]

    # List h5 files
    h5_files = [os.path.join(feats_dir, f) for f in os.listdir(feats_dir) if f.endswith(".h5")]

    # Load model
    model = Classifier(
        dim=classifier_params.get('dim', 512),
        num_heads=classifier_params.get('num_heads', 8),
        num_seeds=classifier_params.get('num_seeds', 4),
        num_classes=len(target_dict)
    )
    model.load_state_dict(torch.load(mil_weights, map_location=device))
    model = model.to(device)
    model.eval()

    results = []

    for path in tqdm(h5_files):
        patient_name = os.path.basename(path).replace(".h5", "")
        patient_id = extract_patient_id(patient_name, fname_index)
        row = clini_df[clini_df["PATIENT"] == patient_id]
        if row.empty:
            continue
        patient_class = row[target_label].iloc[0]
        if patient_class not in target_dict:
            continue
        cls_id = target_dict[patient_class]

        with h5py.File(path, "r") as hdf_file:
            feats = torch.from_numpy(hdf_file["feats"][:]).to(device)
            scores = F.softmax(model(feats.unsqueeze(1)), dim=1)
            num_feats = scores.shape[0]
            k = min(n_top_features, num_feats)
            if k == 0:
                continue
            top_scores, top_indices = scores[:, cls_id].topk(k)
            for idx, score in zip(top_indices.detach().cpu().numpy(), top_scores.detach().cpu().numpy()):
                results.append({
                    "patient_id": patient_id,
                    "h5_file": path,
                    "feature_index": int(idx),
                    "score": float(score),
                    "class": patient_class
                })


    # Save CSV of top features
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved top features to {output_csv}")

    # --- New Step: Extract features and save as .h5 and .csv ---
    if not df.empty:
        feature_list = []
        meta_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features from h5"):
            h5_path = row['h5_file']
            feat_idx = int(row['feature_index'])
            with h5py.File(h5_path, "r") as hdf_file:
                feats = hdf_file["feats"]
                feature_vec = feats[feat_idx]
                feature_list.append(feature_vec)
                meta_list.append({
                    "patient_id": row["patient_id"],
                    "h5_file": h5_path,
                    "feature_index": feat_idx,
                    "score": row["score"],
                    "class": row["class"]
                })
        features_np = np.stack(feature_list)
        # Save as .h5
        features_h5_path = params.get('output_h5', output_csv.replace('.csv', '_features.h5'))
        with h5py.File(features_h5_path, "w") as h5f:
            h5f.create_dataset("features", data=features_np)
        print(f"Saved features array to {features_h5_path}")
        # Save metadata as .csv
        features_meta_csv_path = params.get('output_meta_csv', output_csv.replace('.csv', '_features_meta.csv'))
        pd.DataFrame(meta_list).to_csv(features_meta_csv_path, index=False)
        print(f"Saved features metadata to {features_meta_csv_path}")
    else:
        print("No features to extract.")

if __name__ == "__main__":
    main()