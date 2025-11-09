import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import sys
from collections import defaultdict
from typing import Iterator, Tuple, Union, List


def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    This function operates in two modes:
    1.  If metadata is found, it yields data based on the metadata file.
    2.  If metadata is NOT found, it uses glob to find and yield all '.tsv'
        files in the directory.

    Args:
        data_dir (str): The path to the directory containing the data.

    Yields:
        An iterator of tuples. The format depends on the mode:
        - With metadata: (repertoire_id, pd.DataFrame, label_positive)
        - Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' listed in metadata not found. Skipping.")
                continue
    else:
        # Search recursively so nested structures are supported
        for root, _, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.tsv'):
                    file_path = os.path.join(root, f)
                    try:
                        repertoire_df = pd.read_csv(file_path, sep='\t')
                        yield os.path.basename(file_path), repertoire_df
                    except Exception as e:
                        print(f"Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                        continue


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Loads all TSV files from a directory and concatenates them into a single DataFrame.

    This function handles two scenarios:
    1. If metadata.csv exists, it loads data based on the metadata and adds
       'repertoire_id' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files (recursively)
       and adds a 'filename' column as an identifier.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        # recursive count for progress bar
        total_files = sum(1 for _root, _dirs, files in os.walk(data_dir) for f in files if f.endswith('.tsv'))
        for item in tqdm(data_loader, total=total_files, desc="Loading files"):
            filename, data_df = item
            rep_id = os.path.basename(filename).replace('.tsv', '')
            data_df['ID'] = rep_id
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df


def load_and_encode_kmers(data_dir: str, k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loading and k-mer encoding of repertoire data."""
    from collections import Counter

    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []

    total_files = sum(1 for _root, _dirs, files in os.walk(data_dir) for f in files if f.endswith('.tsv'))

    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace('.tsv', '')
            label = None

        kmer_counts = Counter()
        for seq in data_df.get('junction_aa', pd.Series(dtype=str)).dropna():
            for i in range(len(seq) - k + 1):
                kmer_counts[seq[i:i + k]] += 1

        repertoire_features.append({'ID': rep_id, **kmer_counts})

        meta_rec = {'ID': rep_id}
        if label is not None:
            meta_rec['label_positive'] = label
        metadata_records.append(meta_rec)

    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    features_df = features_df.fillna(0)
    metadata_df = pd.DataFrame(metadata_records)

    return features_df, metadata_df


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_repertoire_ids(data_dir: str) -> list:
    """Retrieves repertoire IDs from metadata or filenames (recursively)."""
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        repertoire_ids = metadata_df['repertoire_id'].tolist()
    else:
        repertoire_ids = []
        for root, _dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.tsv'):
                    repertoire_ids.append(os.path.splitext(f)[0])

    return repertoire_ids


def generate_random_top_sequences_df(n_seq: int = 50000) -> pd.DataFrame:
    """Generates a random DataFrame simulating top important sequences."""
    seqs = set()
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    while len(seqs) < n_seq:
        seq = ''.join(np.random.choice(alphabet, size=15))
        seqs.add(seq)
    data = {
        'junction_aa': list(seqs),
        'v_call': ['TRBV20-1'] * n_seq,
        'j_call': ['TRBJ2-7'] * n_seq,
        'importance_score': np.random.rand(n_seq)
    }
    return pd.DataFrame(data)


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    """Directory validation that supports nested dataset structures."""
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."

    # Recursively ensure there is at least one .tsv somewhere under train_dir
    has_train_tsv = any(
        f.endswith('.tsv')
        for root, _dirs, files in os.walk(train_dir)
        for f in files
    )
    assert has_train_tsv, f"No .tsv files found anywhere under train directory `{train_dir}`."

    # metadata.csv must exist in each train_dataset_* folder (not necessarily at root)
    train_subs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if d.startswith('train_dataset_') and os.path.isdir(os.path.join(train_dir, d))]
    assert train_subs, f"No train_dataset_* folders found under `{train_dir}`."
    missing_meta = [d for d in train_subs if not os.path.isfile(os.path.join(d, 'metadata.csv'))]
    assert not missing_meta, f"Missing metadata.csv in: {missing_meta}"

    # Validate each provided test dir recursively
    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        has_test_tsv = any(
            f.endswith('.tsv')
            for root, _dirs, files in os.walk(test_dir)
            for f in files
        )
        assert has_test_tsv, f"No .tsv files found anywhere under test directory `{test_dir}`."

    # out_dir writability check
    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, "test_write_permission.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """Concatenate per-dataset outputs to submissions.csv (unchanged)."""
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read predictions file '{pred_file}'. Error: {e}. Skipping.")
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read sequences file '{seq_file}'. Error: {e}. Skipping.")
            continue

    if not df_list:
        print("Warning: No output files were found to concatenate.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    if os.path.isdir(test_dir):
        for test_name in sorted(os.listdir(test_dir)):
            tpath = os.path.join(test_dir, test_name)
            if os.path.isdir(tpath) and test_name.startswith("test_dataset_"):
                base_id = test_name.replace("test_dataset_", "").split("_")[0]
                test_groups[base_id].append(tpath)

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        tpath = os.path.join(train_dir, train_name)
        if os.path.isdir(tpath) and train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            pairs.append((tpath, test_groups.get(train_id, [])))

    return pairs
