"""
Deployment script for the NMT system
Loads a trained model and provides interactive translation
"""

import os
import sys
from simple_transformer_nmt import SimpleNMTSystem

def load_model(model_path):
    """Load a trained NMT model"""
    try:
        nmt_system = SimpleNMTSystem()
        nmt_system.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return nmt_system
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def interactive_translation(nmt_system):
    """Interactive translation interface"""
    print("\n" + "="*60)
    print("🌐 English to Bengali Translation Interface")
    print("="*60)
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🇬🇧 Enter English text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n📖 Available commands:")
                print("  - Type any English text to translate")
                print("  - 'quit' or 'exit' to close")
                print("  - 'help' to show this message")
                print("  - 'stats' to show model statistics")
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n📊 Model Statistics:")
                print(f"  - English vocabulary size: {len(nmt_system.english_tokenizer.word2idx)}")
                print(f"  - Bengali vocabulary size: {len(nmt_system.bengali_tokenizer.word2idx)}")
                print(f"  - Max sequence length: {nmt_system.max_length}")
                continue
            
            if not user_input:
                print("⚠️  Please enter some text to translate")
                continue
            
            # Translate
            print("🔄 Translating...")
            translation = nmt_system.predict(user_input)
            
            print(f"🇧🇩 Bengali: {translation}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_translation(nmt_system, input_file, output_file):
    """Translate a file of English sentences"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            english_sentences = [line.strip() for line in f if line.strip()]
        
        print(f"📖 Processing {len(english_sentences)} sentences...")
        
        translations = []
        for i, sentence in enumerate(english_sentences):
            print(f"🔄 Translating {i+1}/{len(english_sentences)}: {sentence[:50]}...")
            translation = nmt_system.predict(sentence)
            translations.append(f"{sentence}\t{translation}")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("English\tBengali\n")
            f.write("\n".join(translations))
        
        print(f"✅ Translations saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error in batch translation: {e}")

def main():
    """Main deployment function"""
    print("🚀 NMT System Deployment")
    print("=" * 40)
    
    # Check if model exists
    model_path = "simple_nmt_model"
    if not os.path.exists(f"{model_path}_model.h5"):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first by running:")
        print("python simple_transformer_nmt.py")
        return
    
    # Load model
    nmt_system = load_model(model_path)
    if nmt_system is None:
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch':
            if len(sys.argv) < 4:
                print("Usage: python deploy_nmt.py --batch <input_file> <output_file>")
                return
            input_file = sys.argv[2]
            output_file = sys.argv[3]
            batch_translation(nmt_system, input_file, output_file)
        else:
            print("Unknown argument. Use --batch for file translation or no arguments for interactive mode.")
            return
    else:
        # Interactive mode
        interactive_translation(nmt_system)

if __name__ == "__main__":
    main()
