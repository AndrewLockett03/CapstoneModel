import kagglehub

# Download latest version
path = kagglehub.dataset_download("pratt3000/vctk-corpus")

print("Path to dataset files:", path)