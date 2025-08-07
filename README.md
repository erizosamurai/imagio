# Imagio: Semantic Image Search CLI

Imagio is a command-line interface (CLI) tool for semantic image search, powered by CLIP (Contrastive Language–Image Pretraining) and Faiss (Facebook AI Similarity Search). It enables users to process a folder of images to generate embeddings, store them in a Faiss index, and search for images using text queries, retrieving the top-k most relevant images above a specified similarity threshold.

---

## Installation

To install Imagio directly on your system:

```bash
git clone https://github.com/erizosamurai/imagio.git
cd imagio
pip install -e .
```

This installs Imagio and its dependencies (`torch`, `transformers`, `Pillow`, `numpy`, `faiss-cpu`). See `requirements.txt` for specific versions.

To verify installation:

```bash
imagio --version
```

Expected output:

```
Imagio version 0.1.0
```

---

## Usage

Imagio provides two main commands: `process` to generate embeddings and `search` to query images.

### Process Command

Generate embeddings for images and save to a Faiss index:

```bash
imagio process --image_folder "data/images" --embeddings_path "embeddings"
```

**Arguments:**

* `--image_folder`: Path to images (e.g., .jpg, .png).
* `--embeddings_path`: Path to save `index.faiss` and `filenames.json`.
* `--model_name` (optional): CLIP model (default: `openai/clip-vit-base-patch32`).
* `--embedding_dim` (optional): Embedding dimension (default: 512).
* `--verbose` (optional): Enable detailed logging.

---

### Search Command

Search for images with a text query:

```bash
imagio search --query "dog" --embeddings_path embeddings --threshold 0.8 --top_k 5
```

**Arguments:**

* `--query`: Text query (e.g., “dog”).
* `--embeddings_path`: Path to `index.faiss` and `filenames.json`.
* `--threshold` (optional): Similarity threshold (default: 0.8).
* `--top_k` (optional): Number of results (default: 5).
* `--model_name` (optional): CLIP model (must match process).
* `--verbose` (optional): Enable detailed logging.

---

## Troubleshooting

* **No images found**: Ensure `--image_folder` contains valid images (.jpg, .png, .jpeg).
* **Faiss index not found**: Run `imagio process` to create `index.faiss` and `filenames.json`.
* **Model download errors**: Check internet connection or specify a valid `--model_name`.
* **No results above threshold**: Lower `--threshold` (e.g., 0.7) or refine the query.

### Faiss OpenMP Error (e.g., “OMP: Error #15”)

**Cause**: Multiple OpenMP runtime libraries (e.g., `libiomp5md.dll` on Windows, `libomp` on Linux/macOS) from `faiss-cpu`, `numpy`, or `torch` are conflicting.

**Temporary Workaround**: Set the environment variable before running Imagio:

* Linux/macOS:

  ```bash
  export KMP_DUPLICATE_LIB_OK=TRUE
  ```
* Windows:

  ```cmd
  set KMP_DUPLICATE_LIB_OK=TRUE
  ```

> Warning: This may cause crashes or incorrect results and is not recommended for production.

**Recommended Solution**: Use conda to install dependencies to avoid conflicts:

```bash
conda install faiss-cpu numpy torch
```

Alternatively, ensure all dependencies use the same OpenMP library.

Report persistent issues at: [https://github.com/erizosamurai/imagio/issues](https://github.com/erizosamurai/imagio/issues)

---

## Contribution Guidelines

I welcome contributions to enhance Imagio. Below are key areas where help is appreciated:

* [ ] **Docstrings**

  * Add detailed docstrings for all functions, classes, and modules using the Google Python Style Guide.
  * Include descriptions, parameters, return values, and exceptions.

* [ ] **Logging**

  * Use the `logging` module for all output, consistent with `utils.py`.
  * Use appropriate logging levels (`debug`, `info`, `warning`, `error`).

* [ ] **Testing**

  * Add unit tests in the `tests/` directory using `unittest` or `pytest`.
  * Cover edge cases (invalid inputs, missing files, OpenMP issues).

* [ ] **Model Compatibility**

  * Add support for advanced models like SigLIP and SigLIP2 (available on Hugging Face).
  * Update `model.py`, `processor.py`, and `search.py` to support new embedding dimensions and dynamic model selection.

* [ ] **Alternative Databases**

  * Explore replacing Faiss with Milvus, Chroma, or Pinecone.
  * Update `processor.py` and `search.py` to use database-specific APIs.
  * Document requirements and ensure functionality with both small and large datasets.

* [ ] **Pull Requests**

  * Submit PRs to the `main` branch with a clear description of changes.
  * Ensure tests pass and include new tests for added functionality.

* [ ] **Licensing**

  * Contributions are licensed under the MIT License.
  * Do not include any proprietary code.

### To Contribute

1. Fork the repository: [https://github.com/erizosamurai/imagio](https://github.com/erizosamurai/imagio)
2. Create a branch:

   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add your-feature"
   ```
4. Push and open a pull request:

   ```bash
   git push origin feature/your-feature
   ```

I especially encourage contributions that:

* Add support for SigLIP or SigLIP2 for better accuracy.
* Integrate alternative databases (Milvus, Chroma, Pinecone) to resolve Faiss OpenMP issues.
* Optimize performance (e.g., Faiss IndexIVFFlat or database-specific indexes).
* Add CLI features (e.g., batch queries, export results to file).

---

## License

Imagio is licensed under the MIT License. See the `LICENSE` file for details.

---

