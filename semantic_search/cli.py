import argparse
import os
import sys

from semantic_search.model import SemanticSearchModel
from semantic_search.processor import ImageProcessor
from semantic_search.search import ImageSearch
from semantic_search.utils import get_image_files

IMAGIO_VERSION = "0.1.0"

def create_parser():
    parser  = argparse.ArgumentParser(
    description="Imagio: A semantic Image search CLI tool using CLIP and FAISS"
    )
    parser.add_argument(
        "--version", action="version", version=f"Imagio version {IMAGIO_VERSION}"
    )

    subparsers = parser.add_subparsers(dest="command",required=True)
    #Process Command
    process_parser = subparsers.add_parser("process",help="Process images to generate Embeddings")
    process_parser.add_argument("--image_folder", required=True, help="Path to the image folder (e.g., data/images/)")
    process_parser.add_argument("--embeddings_path", required=True, help="Path to save the Faiss index and filenames (e.g., embeddings/)")
    process_parser.add_argument("--model_name", default="openai/clip-vit-large-patch14", help="CLIP model name (default:openai/clip-vit-large-patch14)")
    process_parser.add_argument("--embedding_dim", type=int, default=768, help="Embedding dimension (default: 768)")
    process_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # --- Search Command ---
    search_parser = subparsers.add_parser("search", help="Search images by text query")

    search_parser.add_argument("--query", required=True, help="Text query (e.g., 'a dog')")
    search_parser.add_argument("--embeddings_path",help="Path to the Faiss index and filenames")
    search_parser.add_argument("--threshold", type=float, default=50, help="Similarity threshold (default: 0.5)")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of results to return (default: 5)")

    return parser

# ---- Command Implementations ----

def handle_process(args):
    if not os.path.isdir(args.image_folder):
        print(f"[ERROR] Folder not found: {args.image_folder}")
        sys.exit(1)

    image_files = get_image_files(args.image_folder)
    if not image_files:
        print(f"[ERROR] No image files found in folder: {args.image_folder}")
        sys.exit(1)

    os.makedirs(args.embeddings_path, exist_ok=True)

    if args.verbose:
        print(f"[INFO] Found {len(image_files)} image files.")
        print(f"[INFO] Using CLIP model: {args.model_name}")
        print(f"[INFO] Saving embeddings to: {args.embeddings_path}")

    # Replace below with actual processing logic
    print("[INFO] Processing images and generating embeddings... (placeholder)")
    model = SemanticSearchModel(args.model_name)
    processor = ImageProcessor(model=model, path=args.image_folder,dimension=args.embedding_dim)
    embeddings,filenames = processor.process_folder()
    processor.save_embeddings(embeddings=embeddings,file=filenames,save_path=args.embeddings_path)

def handle_search(args):
    if not os.path.isdir(args.embeddings_path):
        print(f"[ERROR] Embeddings path not found: {args.embeddings_path}")
        sys.exit(1)

    print(f"[INFO] Searching for: '{args.query}'")
    print(f"[INFO] Using FAISS index from: {args.embeddings_path}")
    print(f"[INFO] Top-{args.top_k} results with threshold ≥ {args.threshold}")
    search = ImageSearch(path=args.embeddings_path)
    print(search.search(query=args.query,top_k=args.top_k, threshold=args.threshold))


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "process":
        handle_process(args)
    elif args.command == "search":
        handle_search(args)

if __name__ == "__main__":
    main()
