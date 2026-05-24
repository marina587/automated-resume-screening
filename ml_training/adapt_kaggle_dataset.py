import pandas as pd
import numpy as np
import os

def adapt_dataset(input_path, output_path):
    """
    Adapts the Kaggle 'resume_dataset_200k_enhanced.csv' to the format 
    required by the training pipeline (columns: 'resume_text', 'category').
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check required columns exist
    required_cols = ['programming_languages', 'experience_years', 'projects', 
                     'internships', 'skills_score', 'company_type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in source dataset: {missing}")

    print(f"Dataset loaded. Shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")
    
    # 1. Create Synthetic 'resume_text'
    # We combine the structured data into a text blob that looks like a resume summary
    def generate_resume_text(row):
        parts = []
        parts.append(f"Professional with {row['experience_years']} years of experience.")
        
        if isinstance(row['programming_languages'], str):
            parts.append(f"Skilled in {row['programming_languages']}.")
        
        parts.append(f"Completed {row['projects']} major projects and {row['internships']} internships.")
        
        if row.get('certifications') and isinstance(row['certifications'], str):
            parts.append(f"Holds certifications in {row['certifications']}.")
            
        if row.get('research_papers', 0) > 0:
            parts.append(f"Author of {row['research_papers']} research papers.")
            
        parts.append(f"Education: {row.get('education_level', 'Graduate')} from {row.get('university_tier', 'Tier-1')} university with CGPA {row['cgpa']}.")
        
        return " ".join(parts)

    print("Generating synthetic resume text...")
    df['resume_text'] = df.apply(generate_resume_text, axis=1)
    
    # 2. Create 'category' based on 'company_type' or skills
    # Since the dataset doesn't have explicit job titles, we infer them from company_type or assign based on skills
    # Mapping heuristic: If company_type is tech-related, map to Software Engineer, etc.
    # For this dataset, 'company_type' seems to be the closest proxy. Let's map them to standard roles.
    
    unique_companies = df['company_type'].unique()
    print(f"Unique company types found: {unique_companies}")
    
    # Simple mapping logic (adjust based on actual values in your dataset if needed)
    # If company_type is numeric or generic, we might need a different strategy.
    # Assuming company_type contains strings like 'Product Based', 'Service Based', etc.
    # If it's just generic, we will create categories based on skill density or random assignment for demo purposes.
    
    # Strategy: If company_type is not descriptive enough, we create categories based on 'skills_score' and 'experience'
    # OR we simply map the unique company types to job roles if they look like roles.
    
    # Let's inspect the first few values to decide mapping
    print(f"Sample company types: {df['company_type'].head(10).tolist()}")
    
    # Heuristic Mapping (Customize this if your data has specific role names)
    # If 'company_type' is actually just 'MNC', 'Startup', etc., we can't map directly to 'Java Developer'.
    # In that case, we will generate categories based on the dominant programming language.
    
    def infer_category(row):
        # Try to infer from programming languages if possible
        langs = str(row.get('programming_languages', '')).lower()
        if 'python' in langs and ('machine' in str(row.get('projects', '')) or 'data' in str(row.get('projects', ''))):
            return 'Data Scientist'
        elif 'java' in langs or 'spring' in langs:
            return 'Backend Developer'
        elif 'javascript' in langs or 'react' in langs or 'angular' in langs:
            return 'Frontend Developer'
        elif 'devops' in langs or 'aws' in langs or 'docker' in langs:
            return 'DevOps Engineer'
        elif 'sql' in langs and 'analysis' in str(row.get('projects', '')).lower():
            return 'Data Analyst'
        else:
            # Default fallback based on experience or random distribution
            roles = ['Software Engineer', 'Full Stack Developer', 'Backend Developer', 'Data Analyst', 'Product Manager']
            # Deterministic assignment based on candidate_id to ensure balance
            idx = int(row['candidate_id']) % len(roles)
            return roles[idx]

    print("Inferring job categories...")
    df['category'] = df.apply(infer_category, axis=1)
    
    # Select only required columns for training
    final_df = df[['resume_text', 'category']].copy()
    
    # Drop rows where text is empty
    final_df = final_df.dropna(subset=['resume_text', 'category'])
    final_df = final_df[final_df['resume_text'].str.strip() != ""]
    
    print(f"Final adapted dataset shape: {final_df.shape}")
    print(f"Category distribution:\n{final_df['category'].value_counts()}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved adapted dataset to {output_path}")
    return output_path

if __name__ == "__main__":
    input_file = "data/resume_dataset_200k_enhanced.csv"
    output_file = "data/resumes_cleaned.csv" # Standard name expected by trainer
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
    else:
        adapt_dataset(input_file, output_file)
