"""
Script to inspect the Kaggle resume dataset and identify correct column names.
Run this on your local machine where the dataset is located.
"""

import pandas as pd
import sys

def inspect_dataset(file_path):
    """Inspect the dataset and print column information."""
    print(f"\n{'='*60}")
    print(f"Inspecting dataset: {file_path}")
    print(f"{'='*60}\n")
    
    try:
        # Try reading the file
        df = pd.read_csv(file_path)
        
        print(f"✓ Successfully loaded dataset")
        print(f"\n📊 Dataset Info:")
        print(f"   - Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\n📋 Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. '{col}'")
        
        print(f"\n🔍 First 3 rows (sample):")
        print(df.head(3).to_string())
        
        print(f"\n📈 Column Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"   - {col}: {dtype}")
        
        print(f"\n⚠️  Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("   No missing values!")
        else:
            for col, count in missing[missing > 0].items():
                print(f"   - {col}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Try to identify potential text and category columns
        print(f"\n🎯 Auto-Detection Results:")
        
        possible_text_cols = ['resume_text', 'resume', 'text', 'content', 'description', 
                             'resume_content', 'resume_text_cleaned', 'raw_text']
        possible_cat_cols = ['category', 'label', 'class', 'job_title', 'job_category', 
                            'role', 'designation', 'job_role', 'position', 'job_family']
        
        detected_text = [col for col in possible_text_cols if col in df.columns]
        detected_cat = [col for col in possible_cat_cols if col in df.columns]
        
        if detected_text:
            print(f"   ✓ Potential TEXT column(s): {detected_text}")
            for col in detected_text:
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
                print(f"      Sample from '{col}': {str(sample)[:100]}...")
        else:
            print(f"   ⚠ No standard text column found. Check columns manually.")
        
        if detected_cat:
            print(f"   ✓ Potential CATEGORY column(s): {detected_cat}")
            for col in detected_cat:
                unique_vals = df[col].nunique()
                print(f"      '{col}': {unique_vals:,} unique values")
                if unique_vals < 20:
                    print(f"         Values: {df[col].unique()[:10]}")
        else:
            print(f"   ⚠ No standard category column found. Check columns manually.")
        
        print(f"\n{'='*60}")
        print("RECOMMENDED COMMAND:")
        print(f"{'='*60}")
        
        # Suggest the best column names
        text_col = detected_text[0] if detected_text else 'resume_text'
        cat_col = detected_cat[0] if detected_cat else 'category'
        
        print(f"\npython ml_training/train_pipeline.py \\")
        print(f"    --data {file_path} \\")
        print(f"    --model-type logistic \\")
        print(f"    --sample-size 10000")
        
        if text_col != 'resume_text' or cat_col != 'category':
            print(f"\n# If auto-detection fails, you may need to modify the code")
            print(f"# or rename columns in your CSV file")
        
        print(f"\n{'='*60}\n")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{file_path}'")
        print(f"   Please check the path and try again.")
        return False
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    # Default path - change this to your actual path
    default_path = r"C:\Users\user\Downloads\automated-resume-screening\data\resume_dataset_200k_enhanced.csv"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_path
    
    inspect_dataset(file_path)
