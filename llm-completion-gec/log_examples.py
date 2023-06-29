from prompt import generate
from dotenv import load_dotenv
load_dotenv()

EXAMPLE_FILE = "hard_examples.txt"

def main():
    with open(EXAMPLE_FILE, "r") as f:
        examples = f.readlines()

    for example in examples:
        generate(example)

if __name__ == '__main__':
    main()
    
