import glob
import os
import nibabel as nib
import pandas as pd
from nilearn import image, plotting

DATA_DIR = "data/ds000245"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def classify_group(participant_id: str) -> str:
    if "CTL" in participant_id:
        return "Healthy Control (CTL)"
    elif "ODP" in participant_id:
        return "Parkinson - Olfactory Dysfunction (ODP)"
    elif "NDP" in participant_id:
        return "Parkinson - No Olfactory Dysfunction (NDP)"
    return "Unknown"


def inspect_participants(data_dir: str):
    tsv_path = os.path.join(data_dir, "participants.tsv")
    if not os.path.exists(tsv_path):
        print(f"[EROARE] Nu am găsit {tsv_path}. Verifică variabila DATA_DIR!")
        return None

    df = pd.read_csv(tsv_path, sep="\t")
    df["Group"] = df["participant_id"].apply(classify_group)

    print("=== 1. INFORMAȚII PARTICIPANȚI (participants.tsv) ===")
    print(f"Total subiecți în tabel: {len(df)}")
    print(f"Coloane disponibile: {list(df.columns)}\n")
    print("Distribuție pe grupuri:")
    print(df["Group"].value_counts().to_string())
    print("\nPrimele 10 rânduri:")
    print(df.head(10).to_string(index=False))
    return df


def find_resting_state_files(data_dir: str):
    print("\n=== 2. CĂUTARE FIȘIERE fMRI (resting-state) ===")
    # Căutăm în derivatives (dacă există date preprocesate)
    deriv_files = sorted(
        glob.glob(os.path.join(data_dir, "derivatives", "**", "*rest*.nii.gz"), recursive=True)
    )
    # Căutăm în folderele BIDS (sub-CTL*, sub-ODP*, sub-NDP*)
    raw_files = sorted(
        glob.glob(os.path.join(data_dir, "sub-*", "**", "func", "*task-rest*bold.nii.gz"), recursive=True)
    )

    print(f"Fișiere 'rest' găsite în derivatives/: {len(deriv_files)}")
    if deriv_files:
        print(f"  Exemplu: {deriv_files[0]}")

    print(f"Fișiere 'rest' găsite în sub-*/func/: {len(raw_files)}")
    if raw_files:
        print(f"  Exemplu: {raw_files[0]}")

    return deriv_files, raw_files


def inspect_and_plot_subject(nii_path: str, label: str, out_name: str):
    print(f"\n--- Inspectare {label}: {os.path.basename(nii_path)} ---")
    img = nib.load(nii_path)
    shape = img.shape
    zooms = img.header.get_zooms()

    print(f"  Dimensiune matrice (X, Y, Z, Timp): {shape}")
    print(f"  Rezoluție spațială (mm): {zooms[:3]}")
    if len(zooms) > 3:
        print(f"  Repetition Time (TR): {zooms[3]} secunde")

    # Calculăm imaginea medie 3D pe axa timpului și o salvăm
    mean_img = image.mean_img(img)
    out_path = os.path.join(RESULTS_DIR, out_name)
    plotting.plot_epi(
        mean_img,
        title=f"Mean fMRI (Rest) - {label}",
        output_file=out_path,
        display_mode="ortho",
    )
    print(f"  [OK] Imagine salvată în: {out_path}")


if __name__ == "__main__":
    inspect_participants(DATA_DIR)
    deriv_files, raw_files = find_resting_state_files(DATA_DIR)

    files_to_use = deriv_files if len(deriv_files) > 0 else raw_files

    if not files_to_use:
        print("\n[ATENȚIE] Nu am găsit niciun fișier *rest*.nii.gz.")
        print("Rulează comanda de descărcare pentru scanările funcționale din ds000245!")
    else:
        control_files = [f for f in files_to_use if "sub-CTL" in f]
        pd_files = [f for f in files_to_use if ("sub-ODP" in f or "sub-NDP" in f)]

        print(f"\nScanări fMRI Control (CTL): {len(control_files)}")
        print(f"Scanări fMRI Parkinson (ODP + NDP): {len(pd_files)}")

        if control_files:
            inspect_and_plot_subject(control_files[0], "Healthy Control (CTL)", "sample_control.png")
        if pd_files:
            inspect_and_plot_subject(pd_files[0], "Parkinson's Disease (ODP/NDP)", "sample_parkinson.png")