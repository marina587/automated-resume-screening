"""
Data Collection and Preparation Module
Handles loading, exploring, and preprocessing the resume dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


class DataPreparator:
    """Handles data loading, exploration, and initial cleaning."""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load resume dataset from CSV file.
        
        Args:
            file_path: Path to CSV file with resume_text and category columns
            
        Returns:
            DataFrame with resume data
        """
        path = file_path or self.data_path
        if not path:
            raise ValueError("No data path provided")
            
        self.df = pd.read_csv(path)
        print(f"Loaded {len(self.df)} resumes")
        return self.df
    
    def explore_data(self) -> dict:
        """
        Explore dataset for missing values, duplicates, and class balance.
        
        Returns:
            Dictionary with exploration results
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        results = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'class_distribution': self.df['category'].value_counts().to_dict() if 'category' in self.df.columns else {},
            'sample_data': self.df.head(3).to_dict()
        }
        
        print(f"Dataset shape: {results['shape']}")
        print(f"Missing values: {results['missing_values']}")
        print(f"Duplicates: {results['duplicates']}")
        print(f"Class distribution:\n{self.df['category'].value_counts() if 'category' in self.df.columns else 'No category column'}")
        print(f"Available columns: {list(self.df.columns)}")
        
        # Auto-detect potential text and category columns
        possible_text_cols = ['resume_text', 'resume', 'text', 'content', 'description', 'resume_content']
        possible_cat_cols = ['category', 'label', 'class', 'job_title', 'job_category', 'role', 'designation']
        
        detected_text = [col for col in possible_text_cols if col in self.df.columns]
        detected_cat = [col for col in possible_cat_cols if col in self.df.columns]
        
        if detected_text:
            print(f"✓ Detected text column(s): {detected_text}")
        if detected_cat:
            print(f"✓ Detected category column(s): {detected_cat}")
        if not detected_text or not detected_cat:
            print("⚠ WARNING: Could not auto-detect required columns. Please check column names.")
        
        return results
    
    def clean_data(self, category_column: str = None, text_column: str = None) -> pd.DataFrame:
        """
        Remove missing values and duplicates.
        
        Args:
            category_column: Name of the category column (auto-detected if not provided)
            text_column: Name of the resume text column (auto-detected if not provided)
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        # Auto-detect column names if not provided
        if text_column is None:
            # Try common column names for resume text
            possible_text_cols = ['resume_text', 'resume', 'text', 'content', 'description', 'resume_content']
            for col in possible_text_cols:
                if col in self.df.columns:
                    text_column = col
                    break
            if text_column is None:
                raise ValueError(f"Could not find resume text column. Available columns: {list(self.df.columns)}")
        
        if category_column is None:
            # Try common column names for category - expanded list for Kaggle dataset
            possible_cat_cols = ['category', 'label', 'class', 'job_title', 'job_category', 'role', 'designation', 'job_role', 'position']
            for col in possible_cat_cols:
                if col in self.df.columns:
                    category_column = col
                    break
            if category_column is None:
                raise ValueError(f"Could not find category column. Available columns: {list(self.df.columns)}")
        
        print(f"Using text column: '{text_column}', category column: '{category_column}'")
        
        # Rename columns to standard names for consistency
        rename_map = {}
        if text_column != 'resume_text':
            rename_map[text_column] = 'resume_text'
        if category_column != 'category':
            rename_map[category_column] = 'category'
        
        if rename_map:
            self.df = self.df.rename(columns=rename_map)
            print(f"Renamed columns: {rename_map}")
            
        # Convert resume_text to string to handle any non-string values
        self.df['resume_text'] = self.df['resume_text'].astype(str)
        self.df['category'] = self.df['category'].astype(str)
        
        # Remove rows with missing resume_text or category
        self.df = self.df.dropna(subset=['resume_text', 'category'])
        
        # Remove empty strings
        self.df = self.df[self.df['resume_text'].str.strip() != '']
        self.df = self.df[self.df['category'].str.strip() != '']
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['resume_text'])
        
        print(f"Cleaned dataset: {len(self.df)} resumes remaining")
        return self.df
    
    def save_cleaned_data(self, output_path: str = 'data/resumes_cleaned.csv'):
        """Save cleaned data to CSV."""
        if self.df is None:
            raise ValueError("No data to save")
            
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        return output_path


def create_sample_dataset(output_path: str = 'data/sample_resumes.csv', n_samples: int = 100):
    """
    Create a sample dataset for testing when real data is not available.
    
    Args:
        output_path: Path to save the sample dataset
        n_samples: Number of sample resumes to generate
    """
    categories = [
        'Data Scientist', 'Software Engineer', 'Product Manager',
        'UX Designer', 'DevOps Engineer', 'Data Analyst',
        'Machine Learning Engineer', 'Frontend Developer',
        'Backend Developer', 'Full Stack Developer'
    ]
    
    sample_texts = [
        "Experienced data scientist with expertise in machine learning, Python, TensorFlow, and statistical analysis. Led multiple projects in predictive modeling and data visualization.",
        "Software engineer specializing in backend development with Java, Spring Boot, microservices architecture, and cloud technologies including AWS and Docker.",
        "Product manager with 5+ years experience in agile methodologies, user research, product strategy, and cross-functional team leadership.",
        "UX designer proficient in Figma, user research, wireframing, prototyping, and usability testing. Created intuitive interfaces for web and mobile applications.",
        "DevOps engineer skilled in CI/CD pipelines, Kubernetes, Terraform, monitoring systems, and infrastructure automation using Python and Bash scripting.",
        "Data analyst with strong SQL skills, experience in Tableau, Power BI, statistical analysis, and business intelligence reporting.",
        "Machine learning engineer with deep learning expertise, PyTorch, computer vision, NLP, and deployment of ML models at scale.",
        "Frontend developer specializing in React, TypeScript, modern CSS, responsive design, and performance optimization for web applications.",
        "Backend developer with expertise in Node.js, PostgreSQL, Redis, API design, and building scalable distributed systems.",
        "Full stack developer proficient in MERN stack, GraphQL, database design, and end-to-end web application development."
    ]
    
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        category = np.random.choice(categories)
        # Add some variation to the text
        base_text = sample_texts[categories.index(category) % len(sample_texts)]
        variation = f" Candidate ID: {i}. Additional skills include problem-solving, teamwork, and communication. "
        resume_text = base_text + variation + (" " * np.random.randint(0, 50))
        
        data.append({
            'resume_text': resume_text,
            'category': category
        })
    
    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset with {n_samples} resumes at {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage
    sample_path = create_sample_dataset()
    
    preparator = DataPreparator(sample_path)
    preparator.load_data()
    preparator.explore_data()
    preparator.clean_data()
    preparator.save_cleaned_data()
