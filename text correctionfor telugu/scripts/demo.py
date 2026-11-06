
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import TeluguTextCorrectionPipeline


def run_preset_demo(pipeline):
    test_cases = [
        {
            "category": "Test 1: Tenglish Transliteration (Dakshina Lexicon Engine)",
            "input": "namaskaram yela unnaru"
        },
        {
            "category": "Test 2: Telugu Matra & Consonant Spelling Errors",
            "input": "తేలుగు భాశ చాల అందమైణది"
        },
        {
            "category": "Test 3: Practical Academic Sentence with Orthographic Typos",
            "input": "విద్యార్ధులు రేపు పరిక్షలు రాసారు"
        },
        {
            "category": "Test 4: Clean Valid Sentence (Zero False-Positive Check)",
            "input": "భారతదేశం ఒక గొప్ప ప్రజాస్వామ్య దేశం"
        }
    ]
    
    for item in test_cases:
        print(f"\n[{item['category']}]")
        inp = item["input"]
        print(f"  > Input Raw Text  : {inp}")
        
        res = pipeline.correct(inp)
        
        if res.get("was_transliterated"):
            print(f"  > Transliteration : {res['transliterated']}  [Script: Roman -> Telugu]")
            
        print(f"  > NFC Normalized  : {res['normalized']}")
        
        errors = [e['word'] for e in res.get('errors_detected', [])]
        if errors:
            print(f"  > Errors Flagged  : {errors}  [Source: 100K Lexicon OOV]")
        else:
            print(f"  > Errors Flagged  : None (All words verified)")
            
        print(f"  > Final Corrected : {res['corrected']}")
        time.sleep(0.3)

def interactive_mode(pipeline):
    print("\n" + "=" * 72)
    print("   INTERACTIVE TEST MODE (Type any word/sentence, or press Enter to quit)")
    
    while True:
        try:
            user_input = input("\nEnter Telugu or Tenglish text > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive demo. Thank you!")
                break
                
            res = pipeline.correct(user_input)
            print(f"  > Normalized : {res['normalized']}")
            errors = [e['word'] for e in res.get('errors_detected', [])]
            print(f"  > Errors     : {errors if errors else 'None'}")
            print(f"  > Corrected  : {res['corrected']}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":

    print("\nInitializing Pipeline modules (Loading 100K Vocab & Dakshina Lexicon)...")
    pipeline = TeluguTextCorrectionPipeline(mode="hybrid")
    print("Initialization Complete! Running automated test cases...\n")
    
    run_preset_demo(pipeline)

    interactive_mode(pipeline)
