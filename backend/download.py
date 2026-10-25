import io
import os
import urllib.request
import zipfile

from config import DATA_DIR, SUBSETS

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"


def have_all():
    for s in SUBSETS:
        for kind in ["train", "test", "RUL"]:
            if not os.path.exists(os.path.join(DATA_DIR, f"{kind}_{s}.txt")):
                return False
    return True


def extract(z):
    for name in z.namelist():
        base = os.path.basename(name)
        if base.endswith(".zip"):
            extract(zipfile.ZipFile(io.BytesIO(z.read(name))))
        elif base.startswith(("train_FD", "test_FD", "RUL_FD")) and base.endswith(".txt"):
            with open(os.path.join(DATA_DIR, base), "wb") as f:
                f.write(z.read(name))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if have_all():
        print("data already in", DATA_DIR)
        return
    print("downloading", URL)
    try:
        raw = urllib.request.urlopen(URL, timeout=120).read()
        extract(zipfile.ZipFile(io.BytesIO(raw)))
    except Exception as e:
        print("download failed:", e)
    if have_all():
        print("saved to", DATA_DIR)
    else:
        print("could not get the files automatically")
        print("download CMAPSSData.zip from https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
        print("and put train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt in", DATA_DIR)


if __name__ == "__main__":
    main()
