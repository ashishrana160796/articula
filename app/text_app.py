from evaluator.ai_text_detect_spoof import text_detector
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    text_detector(input)
