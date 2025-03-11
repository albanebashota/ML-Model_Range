import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import time
from tqdm import tqdm
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
scaler = joblib.load('scaler.pkl')
stats_df = pd.read_parquet('ml_model.parquet')
session = ort.InferenceSession('label_model.onnx', providers=['CPUExecutionProvider'])
required_features = [
    'neo_make', 'neo_model', 'neo_engine', 'neo_year', 'price', 'miles',
    'Q3_miles', 'Q3_price', 'Q1_price', 'Q1_miles',
    'min_price', 'mean_miles', 'mean_price', 'max_price', 'min_miles', 'max_miles'
]
def get_prediction(batch_df):
    for col in ['neo_make', 'neo_model', 'neo_engine', 'neo_year', 'price', 'miles']:
        if col not in batch_df.columns:
            batch_df[col] = None

    original_price = batch_df['price'].copy()
    original_miles = batch_df['miles'].copy()
    if batch_df['neo_engine'].isnull().all():
        merge_cols = ['neo_make', 'neo_model', 'neo_year']
    elif batch_df['neo_model'].isnull().all():
        merge_cols = ['neo_make', 'neo_year']
    else:
        merge_cols = ['neo_make', 'neo_model', 'neo_year', 'neo_engine']

    batch_merged = batch_df.merge(stats_df, how='left', on=merge_cols)
    batch_merged['price'] = original_price.values
    batch_merged['miles'] = original_miles.values
    cols_to_drop = [
        'price_y', 'msrp_y', 'miles_y', 'state_y', 'dos_active_y',
        'Upper_Bound_miles', 'Lower_Bound_miles',
        'Lower_Bound_price', 'Upper_Bound_price', 'IQR_miles',
        'difference_price', 'difference_miles', 'price_status', 'miles_status',
        'sate', 'dos_active', 'is_outlier'
    ]
    batch_merged.drop(columns=[col for col in cols_to_drop if col in batch_merged.columns], inplace=True)
    for col in ['neo_make', 'neo_model', 'neo_engine']:
        le = label_encoders[col]
        known_classes = dict(zip(le.classes_, le.transform(le.classes_)))
        batch_merged[col] = batch_merged[col].astype(str).map(known_classes)
        batch_merged[col] = batch_merged[col].fillna(-1).astype(int)

    batch_merged.fillna(0, inplace=True)

    missing_cols = [col for col in required_features if col not in batch_merged.columns]
    if missing_cols:
        raise ValueError(f"Mising columns: {missing_cols}")
    X = batch_merged[required_features].astype(float)
    X_scaled = scaler.transform(X)
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
    deal_types = target_encoder.inverse_transform(preds.astype(int))
    batch_merged['deal_type'] = deal_types
    batch_merged['difference_price'] = batch_merged['price'] - batch_merged['mean_price']
    batch_merged['difference_miles'] = batch_merged['miles'] - batch_merged['mean_miles']

    batch_merged['price_status'] = np.where(
        batch_merged['price'] < batch_merged['mean_price'], 'Below Average',
        np.where(batch_merged['price'] > batch_merged['mean_price'], 'Above Average', 'Average')
    )
    batch_merged['miles_status'] = np.where(
        batch_merged['miles'] < batch_merged['mean_miles'], 'Below Average',
        np.where(batch_merged['miles'] > batch_merged['mean_miles'], 'Above Average', 'Average')
    )
    json_cols = ['deal_type', 'min_price', 'max_price', 'Q1_price', 'Q3_price',
                 'difference_price', 'difference_miles', 'price_status', 'miles_status']
    deal_type_json = batch_merged[json_cols].apply(lambda row: json.dumps(row.to_dict()), axis=1)

    return deal_type_json

def process_file(input_file, output_file, chunksize=50000):
    first = True
    total_start = time.time()

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize, compression='gzip')):
        batch_start = time.time()
        print(f"Batch #{i+1} - Rreshta: {len(chunk)}")
        chunk['deal_type'] = get_prediction(chunk)

        if first:
            chunk.to_csv(output_file, index=False, mode='w', compression='gzip')
            first = False
        else:
            chunk.to_csv(output_file, index=False, header=False, mode='a', compression='gzip')

        print(f"Batch #{i+1} finished {round(time.time() - batch_start, 2)} sekonda.")

    print(f"Finished in {input_file} → {output_file} total {round(time.time() - total_start, 2)} sekonda.")

def main():
    process_file(
        "20250310_processed_used_cars.csv.gz",
        "20250310_predictions_used.csv.gz"
    )
    process_file(
        "20250310_processed_new_cars.csv.gz",
        "20250310_predictions_new.csv.gz"
    )
if __name__ == "__main__":
    main()
