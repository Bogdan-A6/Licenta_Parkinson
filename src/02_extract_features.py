import glob
import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

DATA_DIR = "data/ds000245"
RESULTS_DIR = "results"
OUTPUT_CSV = "data/features_connectivity.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_label_and_group(participant_id: str):
    if "CTL" in participant_id:
        return 0, "CTL"
    elif "ODP" in participant_id:
        return 1, "ODP"
    elif "ODN" in participant_id or "NDP" in participant_id:
        return 1, "ODN"
    return -1, "Unknown"


def main():
    # 1. Încărcăm metadatele din participants.tsv
    tsv_path = os.path.join(DATA_DIR, "participants.tsv")
    participants_df = pd.read_csv(tsv_path, sep="\t")

    # 2. Descărcăm/încărcăm atlasul funcțional MSDL (39 de regiuni cerebrale)
    print("Se încarcă atlasul cerebral MSDL...")
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas["maps"]
    labels = atlas["labels"]
    print(f"Atlas încărcat cu {len(labels)} regiuni cerebrale.\n")

    # Inițializăm extractorul de semnal + preprocesarea (detrend, standardize, band-pass)
    # Notă: t_r va fi setat dinamic la fiecare subiect din header-ul NIfTI
    connectivity_measure = ConnectivityMeasure(
        kind="correlation",
        vectorize=True,
        discard_diagonal=True,
    )
    connectivity_matrix_measure = ConnectivityMeasure(
        kind="correlation",
        vectorize=False,
    )

    records = []
    ctl_matrices = []
    pd_matrices = []

    # 3. Iterăm prin toți subiecții din participants.tsv
    for idx, row in participants_df.iterrows():
        sub_id = row["participant_id"]
        target, subgroup = get_label_and_group(sub_id)
        if target == -1:
            continue

        # Căutăm fișierul functional resting-state al subiectului
        pattern = os.path.join(DATA_DIR, sub_id, "**", "func", "*task-rest*bold.nii.gz")
        files = sorted(glob.glob(pattern, recursive=True))

        if not files:
            print(f"[SKIP] Lipsă fișier fMRI pentru {sub_id}")
            continue

        nii_path = files[0]
        img = nib.load(nii_path)
        zooms = img.header.get_zooms()
        tr = float(zooms[3]) if len(zooms) > 3 else 2.0

        print(f"[{idx+1}/{len(participants_df)}] Procesez {sub_id} ({subgroup}) | Shape: {img.shape} | TR: {tr}s")

        # Extragem seriile temporale curate (0.01 - 0.1 Hz este standardul pentru resting-state fMRI)
        masker = NiftiMapsMasker(
            maps_img=atlas_filename,
            standardize="zscore_sample",
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=tr,
            memory="nilearn_cache",
            verbose=0,
        )
        time_series = masker.fit_transform(nii_path)

        # Calculăm vectorul de conectivitate (741 features) și matricea 39x39
        conn_vec = connectivity_measure.fit_transform([time_series])[0]
        conn_mat = connectivity_matrix_measure.fit_transform([time_series])[0]

        if target == 0:
            ctl_matrices.append(conn_mat)
        else:
            pd_matrices.append(conn_mat)

        # Construim rândul pentru dataset-ul final
        record = {
            "participant_id": sub_id,
            "target": target,          # 0 = Healthy Control, 1 = Parkinson
            "subgroup": subgroup,      # CTL, ODP, ODN
            "Age": row.get("Age", np.nan),
            "Gender": row.get("Gender", np.nan),
            "ACER": row.get("ACER", np.nan),
            "MMSE": row.get("MMSE", np.nan),
            "OSITJ": row.get("OSITJ", np.nan),
        }
        for i, val in enumerate(conn_vec):
            record[f"conn_{i}"] = val

        records.append(record)

    # 4. Salvăm CSV-ul final cu trăsături
    features_df = pd.DataFrame(records)
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCES] Am salvat {len(features_df)} subiecți și {len(conn_vec)} trăsături fMRI în '{OUTPUT_CSV}'!")
    print("Distribuție clase salvate:")
    print(features_df["subgroup"].value_counts().to_string())

    # 5. Generăm vizualizarea comparativă (Control vs Parkinson)
    if ctl_matrices and pd_matrices:
        mean_ctl = np.mean(ctl_matrices, axis=0)
        mean_pd = np.mean(pd_matrices, axis=0)
        diff_mat = mean_pd - mean_ctl

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plotting.plot_matrix(
            mean_ctl, labels=labels, colorbar=True, vmax=0.8, vmin=-0.8,
            axes=axes[0], title="Conectivitate Medie - Control (CTL)"
        )
        plotting.plot_matrix(
            mean_pd, labels=labels, colorbar=True, vmax=0.8, vmin=-0.8,
            axes=axes[1], title="Conectivitate Medie - Parkinson (PD)"
        )
        plotting.plot_matrix(
            diff_mat, labels=labels, colorbar=True, vmax=0.3, vmin=-0.3,
            axes=axes[2], title="Diferență (Parkinson - Control)"
        )
        plt.tight_layout()
        comp_path = os.path.join(RESULTS_DIR, "connectivity_comparison.png")
        plt.savefig(comp_path, dpi=200)
        print(f"[OK] Grafic comparativ salvat în: {comp_path}")


if __name__ == "__main__":
    main()