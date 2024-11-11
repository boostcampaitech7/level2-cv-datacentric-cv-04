import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data', help='Input folder containing images')
    parser.add_argument('-o', '--output', type=str, default='../up2x_results', help='Output folder')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='Upscaling factor')
    parser.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU device to use')
    parser.add_argument('-t', '--tile', type=int, default=512, help='Tile size for processing images (0 for no tile)')

    args = parser.parse_args()
    data_path = "../"

    # Define input folders
    input_folders = [
        'data/chinese_receipt/img/test',
        'data/chinese_receipt/img/train',
        'data/japanese_receipt/img/test',
        'data/japanese_receipt/img/train',
        'data/thai_receipt/img/test',
        'data/thai_receipt/img/train',
        'data/vietnamese_receipt/img/test',
        'data/vietnamese_receipt/img/train',
    ]


    # Process each input folder
    for input_folder in input_folders:
        # Construct the command to run inference_realesrgan.py
        command = [
            'python', 'inference_realesrgan.py',
            '-i', os.path.join(data_path, input_folder),
            '-o', os.path.join(args.output, input_folder),
            '-n', args.model_name,
            '-s', str(args.outscale),
            # '-g', str(args.gpu_id) if args.gpu_id is not None else '',
            '-t', str(args.tile),
        ]
        
        # Run the command
        subprocess.run(command)

if __name__ == '__main__':
    main()
