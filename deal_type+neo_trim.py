import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort
import json
import time

# Load models and stats
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
scaler = joblib.load('scaler.pkl')
stats_df = pd.read_parquet('ml_model.parquet')
zone_encoders = joblib.load('zone_encoders.pkl')
session = ort.InferenceSession('label_model.onnx', providers=['CPUExecutionProvider'])

required_features = [
    'neo_make', 'neo_model', 'neo_engine', 'neo_trim', 'neo_year', 'price', 'miles',
    'Q3_miles', 'Q3_price', 'Q1_price', 'Q1_miles',
    'min_price', 'mean_miles', 'mean_price',
    'max_price', 'min_miles', 'max_miles',
    'price_zone_enc', 'miles_zone_enc',
    'price_to_mean', 'price_diff_from_mean',
    'miles_to_mean', 'miles_diff_from_mean'
]

stat_cols = [
    "min_price", "Q1_price", "mean_price", "Q3_price", "max_price",
    "min_miles", "Q1_miles", "mean_miles", "Q3_miles", "max_miles"
]

not_found = []

def merge_stats_fast(batch_df):
    keys = ['neo_make', 'neo_model', 'neo_year', 'neo_engine', 'neo_trim']
    for col in keys:
        batch_df[col] = batch_df[col].fillna('').astype(str)
        stats_df[col] = stats_df[col].fillna('').astype(str)
    merged_df = pd.merge(batch_df, stats_df, on=keys, how='left')
    return merged_df

def assign_zone_vectorized(df, value_col, prefix):
    conds = [
        df[value_col] < df[f'min_{prefix}'],
        (df[value_col] >= df[f'min_{prefix}']) & (df[value_col] < df[f'Q1_{prefix}']),
        (df[value_col] >= df[f'Q1_{prefix}']) & (df[value_col] < df[f'mean_{prefix}']),
        (df[value_col] >= df[f'mean_{prefix}']) & (df[value_col] <= df[f'Q3_{prefix}']),
        (df[value_col] > df[f'Q3_{prefix}']) & (df[value_col] <= df[f'max_{prefix}']),
        df[value_col] > df[f'max_{prefix}']
    ]
    choices = ['Extreme Low', 'Very Low', 'Low', 'Mid', 'High', 'Very High']
    return np.select(conds, choices, default=None)

def classify_by_matrix(row):
    price = row["price"]
    miles = row["miles"]
    if price == 0 or pd.isnull(price) or pd.isnull(miles):
        return "No Rating"
    if any(pd.isnull(row[col]) for col in stat_cols):
        return "Uncertain"
    return None

def get_prediction(batch_df):
    df_original = batch_df.copy().reset_index(drop=True)
    original_price = df_original['price'].copy()
    original_miles = df_original['miles'].copy()

    batch_df = merge_stats_fast(df_original)
    batch_df['price'] = original_price
    batch_df['miles'] = original_miles
    batch_df = batch_df.reset_index(drop=True)

    # Track not found combinations
    not_found_rows = batch_df[batch_df['mean_price'].isna()]
    if not not_found_rows.empty:
        global not_found
        not_found += not_found_rows[['neo_make', 'neo_model', 'neo_year', 'neo_engine', 'neo_trim']].to_dict(orient='records')

    # Classify "No Rating" and "Uncertain"
    batch_df["true_deal_type"] = batch_df.apply(classify_by_matrix, axis=1)

    # Split rows to predict and labeled directly
    to_predict_df = batch_df[batch_df["true_deal_type"].isnull()].copy()
    labeled_df = batch_df[batch_df["true_deal_type"].notnull()].copy()

    if not to_predict_df.empty:
        # Encode categorical
        for col in ['neo_make', 'neo_model', 'neo_engine', 'neo_trim']:
            le = label_encoders[col]
            classes = set(le.classes_)
            to_predict_df[col] = to_predict_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in classes else -1)

        # Assign zones
        to_predict_df['price_zone'] = assign_zone_vectorized(to_predict_df, 'price', 'price')
        to_predict_df['miles_zone'] = assign_zone_vectorized(to_predict_df, 'miles', 'miles')

        for col in ['price_zone', 'miles_zone']:
            le = zone_encoders[col]
            classes = set(le.classes_)
            to_predict_df[col + '_enc'] = to_predict_df[col].map(lambda x: le.transform([x])[0] if x in classes else -1)

        # Feature engineering
        to_predict_df['price_to_mean'] = to_predict_df['price'] / to_predict_df['mean_price']
        to_predict_df['price_diff_from_mean'] = to_predict_df['price'] - to_predict_df['mean_price']
        to_predict_df['miles_to_mean'] = to_predict_df['miles'] / to_predict_df['mean_miles']
        to_predict_df['miles_diff_from_mean'] = to_predict_df['miles'] - to_predict_df['mean_miles']

        # Fill NaNs and predict
        to_predict_df[required_features] = to_predict_df[required_features].fillna(0).astype(np.float32)
        X_scaled = scaler.transform(to_predict_df[required_features])
        input_name = session.get_inputs()[0].name
        preds = session.run(None, {input_name: X_scaled})[0]
        deal_types = target_encoder.inverse_transform(preds.astype(int))
        to_predict_df['true_deal_type'] = deal_types

    # Combine all rows
    batch_df = pd.concat([to_predict_df, labeled_df], ignore_index=True)

    # Additional calculated fields
    batch_df['difference_price'] = batch_df['price'] - batch_df['mean_price']
    batch_df['difference_miles'] = batch_df['miles'] - batch_df['mean_miles']
    batch_df['price_status'] = np.where(
        batch_df['price'] < batch_df['mean_price'], 'Below Average',
        np.where(batch_df['price'] > batch_df['mean_price'], 'Above Average', 'Average')
    )
    batch_df['miles_status'] = np.where(
        batch_df['miles'] < batch_df['mean_miles'], 'Below Average',
        np.where(batch_df['miles'] > batch_df['mean_miles'], 'Above Average', 'Average')
    )

    # Build JSON deal_type
    def build_json(row):
        if row['true_deal_type'] in ['Uncertain', 'No Rating']:
            return json.dumps({'deal_type': row['true_deal_type']})
        return json.dumps({
            'deal_type': row['true_deal_type'],
            'min_price': row['min_price'],
            'max_price': row['max_price'],
            'Q1_price': row['Q1_price'],
            'Q3_price': row['Q3_price'],
            'difference_price': row['difference_price'],
            'difference_miles': row['difference_miles'],
            'price_status': row['price_status'],
            'miles_status': row['miles_status']
        })

    batch_df['deal_type'] = batch_df.apply(build_json, axis=1)
    assert len(batch_df) == len(df_original), "Mismatch in row count between input and output!"

    return batch_df[['deal_type']].reset_index(drop=True)

def process_file(input_file, output_file, chunksize=50000):
    first = True
    total_start = time.time()

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize, compression='infer')):
        batch_start = time.time()
        print(f"Batch #{i+1} - Rreshta: {len(chunk)}")

        chunk = chunk.reset_index(drop=True)
        result = get_prediction(chunk)
        result = result.reset_index(drop=True)
        chunk['deal_type'] = result['deal_type']

        try:
            compression = 'gzip' if output_file.endswith('.gz') else None
            if first:
                chunk.to_csv(output_file, index=False, mode='w', compression=compression)
                first = False
            else:
                chunk.to_csv(output_file, index=False, header=False, mode='a', compression=compression)
        except Exception as e:
            print(f"[ERROR] while saving in batches #{i+1}: {e}")

        print(f"Batch #{i+1} finished in {round(time.time() - batch_start, 2)} seconds.")

    print(f"Total time {input_file} → {output_file}: {round(time.time() - total_start, 2)} seconds.")

def main():
    process_file("20250421_processed_used_cars.csv.gz", "20250421_predictions_used_cars.csv.gz")
    process_file("20250421_processed_new_cars.csv.gz", "20250421_predictions_new_cars.csv.gz")

    if not_found:
        nf_df = pd.DataFrame(not_found).drop_duplicates()
        nf_df.to_csv("not_found_stats.csv", index=False)
        print(f"[INFO] {len(nf_df)} combinations not founded and saved in not_found_stats.csv")

if __name__ == "__main__":
    main()
