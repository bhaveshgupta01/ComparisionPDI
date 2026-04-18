# Dataset

## MolsTrans

- `dataset/MolsTrans` is split into three files: `train.csv`, `val.csv`, and `test.csv`. This dataset was curated by [MolTrans](https://github.com/kexinhuang12345/MolTrans).
- The prediction value is binary (`0` or `1`), indicating whether the drug interacts with the protein or not.

---

## BindingDB

- [Download link for BindingDB](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)

| File | Compressed Download | Size on Disk | Last Updated |
|---|---|---|---|
| `BindingDB_BindingDB_Articles.tsv` | `BindingDB_BindingDB_Articles_202604_tsv.zip` | ~330 MB | 2026-03-31 |
| `BindingDB_PDSPKi.tsv` | `BindingDB_PDSPKi_202604_tsv.zip` | ~64 MB | 2026-03-31 |

### ⚠️ Git Large File Storage (Git LFS)

Both BindingDB `.tsv` files exceed GitHub's 100 MB file size limit and are therefore tracked using **[Git LFS](https://git-lfs.com/)**.

After cloning the repository, you **must** install Git LFS and pull the large files separately, or the `.tsv` files will only contain LFS pointer stubs instead of the actual data.

#### Prerequisites

Install Git LFS (one-time, per machine):

```bash
# macOS (Homebrew)
brew install git-lfs

# Ubuntu / Debian
sudo apt-get install git-lfs

# Then initialise LFS for your Git user (one-time)
git lfs install
```

#### Cloning the repository

```bash
# Clone normally — LFS pointers are fetched automatically if git-lfs is installed
git clone <repo-url>
cd ComparisionPDI

# If git-lfs was installed AFTER cloning, pull the actual files manually
git lfs pull
```

#### Verifying LFS files are present

```bash
# Should show the real file size (~330 MB), not a tiny pointer (~130 B)
ls -lh dataset/BindingDB/BindingDB_BindingDB_Articles.tsv

# Or check LFS tracking status
git lfs ls-files
```

#### LFS tracking configuration

The following pattern is recorded in `.gitattributes` to ensure all current and future `.tsv` files inside `dataset/BindingDB/` are tracked by LFS:

```
dataset/BindingDB/*.tsv filter=lfs diff=lfs merge=lfs -text
```
