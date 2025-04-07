import sys
import argparse
from program.models.traditional import run_traditional_compression
from program.models.vp_tree import run_VPtree_compression
from program.models.kd_only import run_kd_only_compression
from program.models.enhanced import run_enhanced_compression
from program.CNN_model import train_cnn_model
from program.preprocess_module import preprocess_images


# Main function to parse arguments and execute selected part
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fractal, CNN, or hybrid compression"
    )
    
    # Argument for choosing which part of the pipeline to run
    parser.add_argument(
        "--part",
        choices=["traditional", "kd_tree", "vp_tree", "enhanced", "preprocess", "train"],
        required=True,
        help="Select the pipeline part to run: train, preprocess, traditional, vp-tree + CNN, or enhanced compression"
    )
    
    # Arguments for fractal and CNN compression
    parser.add_argument(
        "--original_path",
        type=str,
        default=None,
        help="Path to the original image folder"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/compressed/test",
        help="Path to the compressed image folder"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images to process (must be a positive integer)"
    )

    # Argument for evaluation
    args = parser.parse_args()

    # Run the specified part of the pipeline
    if args.part == "traditional":
        if not args.original_path:
            print("Error: --original_path is required for fractal compression.")
            sys.exit(1)
        run_traditional_compression(args.original_path, args.output_path+"/traditional",  limit=args.limit)

    elif args.part == "vp_tree":
        if not args.original_path:
            print("Error: --original_path is required for the VP-tree + CNN compression.")
            sys.exit(1)
        run_VPtree_compression(args.original_path, args.output_path+"/vp_tree", limit=args.limit)

    elif args.part == "kd_tree":
        if not args.original_path:
            print("Error: --original_path is required for the KD-tree only compression.")
            sys.exit(1)
        run_kd_only_compression(args.original_path, args.output_path+"/kd_tree", limit=args.limit)

    elif args.part == "enhanced":
        if not args.original_path:
            print("Error: --original_path is required for the KD-tree + CNN compression.")
            sys.exit(1)
        run_enhanced_compression(args.original_path, args.output_path+"/enhanced", limit=args.limit)

    elif args.part == "preprocess":
        if not args.original_path:
            print("Error: --original_path is required to preprocess the data.")
            sys.exit(1)
        preprocess_images(args.original_path, "data/preprocessed", limit=args.limit)
    
    elif args.part == "train":
        if not args.original_path:
            print("Error: --original_path is required to train the model.")
            sys.exit(1)
        train_cnn_model(args.original_path, args.output_path)
    


#   data_collection

#   traditional                                       "data/dataset/pituitary"   --limit = no. of data to be use (remove if ALL)
#   python main.py --part traditional --original_path "data/dataset/glioma" --limit 1

#   vp-tree                                       "data/dataset/pituitary"
#   python main.py --part vp_tree --original_path "data/dataset/glioma" --limit 5

#   kd-tree only                                  "data/dataset/pituitary"
#   python main.py --part kd_tree --original_path "data/dataset/glioma" --limit 5

#   enhanced                                       "data/dataset/pituitary"
#   python main.py --part enhanced --original_path "data/dataset/glioma" --limit 5

#   preprocessing                    --original_path ay pwede na "data/dataset/glioma" or "data/dataset/pituitary"
#   python main.py --part preprocess --original_path "data/dataset/pituitary" --limit 10

#   cnn model training
#   python main.py --part train --original_path data/preprocessed --output_path data/features
